import random
import torch
from models.quantization import quan_Conv2d, quan_Linear, quantize
import operator
from attack.data_conversion import *


class DualModelBFA(object):
    def __init__(self, criterion, model, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss_studnet = 0
        self.loss_teacher = 0
        self.loss =0
        self.SOURCE_LOSS_CONSTRAINT = 0.1
        
        # attributes for random attack
        self.module_list = []
        for name, m in model.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                self.module_list.append(name)       

    def flip_bit(self, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        if self.k_top is None:
            k_top = m.weight.detach().flatten().__len__()
        else: 
            k_top = self.k_top
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]
        
        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data
    
        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,k_top).short()) \
        // m.b_w.abs().repeat(1,k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass
        
        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk

        
        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        param_flipped = bin2int(w_bin,
                                m.N_bits).view(m.weight.data.size()).float()
        return param_flipped

    def dual_model_progressive_bit_search(self, feature_extractor, classifier_student, classifier_teacher, data_student, data_teacher, target_student, target_teacher):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        feature_extractor.eval()
        classifier_teacher.eval()
        classifier_student.eval()

        # 1. perform the inference w.r.t given data and target

        feature_student = feature_extractor(data_student)
        feature_student = feature_student.view(feature_student.size(0), -1)
        output_student = classifier_student(feature_student)

        feature_teacher = feature_extractor(data_teacher)
        feature_teacher = feature_teacher.view(feature_teacher.size(0), -1)
        output_teacher = classifier_teacher(feature_teacher)

        #         _, target = output.data.max(1)
        self.loss_student = self.criterion(output_student, target_student)
        self.loss_teacher = self.criterion(output_teacher, target_teacher)

        # 2. zero out the grads first, then get the grads
        for m in feature_extractor.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        alpha = 1
        beta  = 1
        self.loss_total = beta * self.loss_student - alpha * self.loss_teacher
        self.loss_total.backward()

        # init the loss_max to enable the while loop

        self.loss_max = beta * self.loss_student.item() - alpha * self.loss_teacher.item()

        # for m in feature_extractor.modules():
        #     if hasattr(m, 'weight'):
        #         if m.weight.grad is not None:
        #             print(m.weight.grad.data)
        #             break

        # 3. for each layer flip #bits = self.bits2flip
        while self.loss_max <= beta * self.loss_student.item() - alpha * self.loss_teacher.item():
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            for name, module in feature_extractor.named_modules():
                if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                    clean_weight = module.weight.data.detach()
                    attack_weight = self.flip_bit(module)

                    # change the weight to attacked weight and get loss
                    module.weight.data = attack_weight
                    feature_student = feature_extractor(data_student)
                    feature_student = feature_student.view(feature_student.size(0), -1)
                    output_student = classifier_student(feature_student)

                    feature_teacher = feature_extractor(data_teacher)
                    feature_teacher = feature_teacher.view(feature_teacher.size(0), -1)
                    output_teacher = classifier_teacher(feature_teacher)

                    # If the source loss exceeds the constraint, undo the bit flip
                    # source_loss = self.criterion(output_student, target_student).item()
                    # if source_loss > self.SOURCE_LOSS_CONSTRAINT:
                    #     module.weight.data = clean_weight  # undo bit flip
                    #     continue  # skip to the next bit

                    self.loss_dict[name] = beta * self.criterion(output_student, target_student).item() \
                        - alpha * self.criterion(output_teacher, target_teacher).item()

                    # change the weight back to the clean weight
                    module.weight.data = clean_weight
            

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = max(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change that layer's weight without putting back the clean weight
        for module_idx, (name, module) in enumerate(feature_extractor.named_modules()):
            if name == max_loss_module:
                # print(name, self.loss.item(), loss_max)
                attack_weight = self.flip_bit(module)
                
                ###########################################################
                ## Attack profiling
                #############################################
                weight_mismatch = attack_weight - module.weight.detach()
                attack_weight_idx = torch.nonzero(weight_mismatch)
                print(attack_weight_idx.shape)
                print('attacked module:', max_loss_module)
                
                attack_log = [] # init an empty list for profile
                
                for i in range(attack_weight_idx.size()[0]):
                    
                    weight_idx = attack_weight_idx[i,:].cpu().numpy()
                    weight_prior = module.weight.detach()[tuple(attack_weight_idx[i,:])].item()
                    weight_post = attack_weight[tuple(attack_weight_idx[i,:])].item()
                    
                    print('attacked weight index:', weight_idx)
                    print('weight before attack:', weight_prior)
                    print('weight after attack:', weight_post)
                    
                    tmp_list = [module_idx, # module index in the net
                                self.bit_counter + (i+1), # current bit-flip index
                                max_loss_module, # current bit-flip module
                                weight_idx, # attacked weight index in weight tensor
                                weight_prior, # weight magnitude before attack
                                weight_post # weight magnitude after attack
                                ] 
                    attack_log.append(tmp_list)

                ###############################################################    
                
                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return attack_log


    def progressive_bit_search(self, model, data, target):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data)
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target)
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        for m in model.modules():
            if hasattr(m, 'weight'):
                if m.weight.grad is not None:
                    print(m.weight.grad.data)

        # 3. for each layer flip #bits = self.bits2flip
        while self.loss_max <= self.loss.item():

            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            for name, module in model.named_modules():
                if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                    clean_weight = module.weight.data.detach()
                    attack_weight = self.flip_bit(module)
                    
                    # change the weight to attacked weight and get loss
                    module.weight.data = attack_weight
                    output = model(data)
                    self.loss_dict[name] = self.criterion(output,
                                                          target).item()
                    # change the weight back to the clean weight
                    module.weight.data = clean_weight
            
            # after going through all the layer, now we find the layer with max loss
            max_loss_module = max(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change that layer's weight without putting back the clean weight
        for module_idx, (name, module) in enumerate(model.named_modules()):
            if name == max_loss_module:
                # print(name, self.loss.item(), loss_max)
                attack_weight = self.flip_bit(module)
                
                ###########################################################
                ## Attack profiling
                #############################################
                weight_mismatch = attack_weight - module.weight.detach()
                attack_weight_idx = torch.nonzero(weight_mismatch)
                print(attack_weight_idx.shape)
                print('attacked module:', max_loss_module)
                
                attack_log = [] # init an empty list for profile
                
                for i in range(attack_weight_idx.size()[0]):
                    
                    weight_idx = attack_weight_idx[i,:].cpu().numpy()
                    weight_prior = module.weight.detach()[tuple(attack_weight_idx[i,:])].item()
                    weight_post = attack_weight[tuple(attack_weight_idx[i,:])].item()
                    
                    print('attacked weight index:', weight_idx)
                    print('weight before attack:', weight_prior)
                    print('weight after attack:', weight_post)
                    
                    tmp_list = [module_idx, # module index in the net
                                self.bit_counter + (i+1), # current bit-flip index
                                max_loss_module, # current bit-flip module
                                weight_idx, # attacked weight index in weight tensor
                                weight_prior, # weight magnitude before attack
                                weight_post # weight magnitude after attack
                                ] 
                    attack_log.append(tmp_list)

                ###############################################################    
                
                
                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return attack_log


    def random_flip_one_bit(self, model):
        """
        Note that, the random bit-flip may not support on binary weight quantization.
        """
        chosen_module = random.choice(self.module_list)
        for name, m in model.named_modules():
            if name == chosen_module:
                flatten_weight = m.weight.detach().view(-1)
                chosen_idx = random.choice(range(flatten_weight.__len__()))
                # convert the chosen weight to 2's complement
                bin_w = int2bin(flatten_weight[chosen_idx], m.N_bits).short()
                # randomly select one bit
                bit_idx = random.choice(range(m.N_bits))
                mask = (bin_w.clone().zero_() + 1) * (2**bit_idx)
                bin_w = bin_w ^ mask
                int_w = bin2int(bin_w, m.N_bits).float()
                
                ##############################################
                ###   attack profiling
                ###############################################
                
                weight_mismatch = flatten_weight[chosen_idx] - int_w
                attack_weight_idx = chosen_idx
                
                print('attacked module:', chosen_module)
                
                attack_log = [] # init an empty list for profile
                
                
                weight_idx = chosen_idx
                weight_prior = flatten_weight[chosen_idx]
                weight_post = int_w

                print('attacked weight index:', weight_idx)
                print('weight before attack:', weight_prior)
                print('weight after attack:', weight_post)  
                
                tmp_list = ["module_idx", # module index in the net
                            self.bit_counter + 1, # current bit-flip index
                            "loss", # current bit-flip module
                            weight_idx, # attacked weight index in weight tensor
                            weight_prior, # weight magnitude before attack
                            weight_post # weight magnitude after attack
                            ] 
                attack_log.append(tmp_list)                            
                
                self.bit_counter += 1
                #################################
                
                flatten_weight[chosen_idx] = int_w
                m.weight.data = flatten_weight.view(m.weight.data.size())
                
            
                
        return attack_log
