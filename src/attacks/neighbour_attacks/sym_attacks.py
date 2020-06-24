import numpy as np
import torch.nn.functional as F
from .utils import (softmax, prob_normalize)



class RGA():
    """
        Reinforced Genetic Attack
    """
    def __init__(self, model, LS,
                 pop_size=20, max_iters=5,
                 top_n=10, packed=True, filter_spec=False,
                 greedy=False, targeted=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.LS = LS
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.top_n = top_n  # similar words
        self.packed = packed
        self.filter_spec = filter_spec
        self.greedy = greedy
        self.targeted = targeted
        self.device = device

    def prepare_batch(self, xs):
        x_in = torch.from_numpy(xs).permute(1,0).to(self.device)
        if self.packed:
            length = x_in.shape[0]
            N = x_in.shape[1]
            return x_in, torch.tensor(length, device=self.device).repeat(N)
        else:
            return x_in

    def to_numpy(self, tensor):
        return tensor.cpu().numpy()

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_replacement(self, pos, x_cur, x_orig, target, subs):
        new_xs = [self.do_replace(x_cur, pos, w)
                  if x_orig[pos] != w and w != 0 \
                  else x_cur \
                  for w in subs]
        
        batch = self.prepare_batch(np.array(new_xs))
        new_preds = self.model.predict_proba(batch)
        new_scores = new_preds[:, target]

        if not self.targeted:
            new_scores = 1. - new_scores

        # For greedy approach.
        # batch = self.prepare_batch(x_cur[np.newaxis, :])
        # orig_score = self.model.predict_proba(batch)[0, target]
        # new_x_scores = new_x_scores - orig_score
        # new_x_scores = self.to_numpy(new_x_scores)

        if self.greedy:
            idx = torch.argmax(new_scores)
        else:
            torch_probs = F.softmax(new_scores, dim=0)
            probs = self.to_numpy(torch_probs)
            idx = np.random.choice(len(new_xs), size=1, p=probs)[0]
        
        return new_xs[idx]

    def perturb(self, x_cur, x_orig, nghbrs, probs, target):
        x_len = probs.shape[0]
        idx = np.random.choice(x_len, size=1, p=probs)[0]
        subs = nghbrs[idx]
        
        if subs.size == 0:
            return x_cur

        return self.select_replacement(idx, x_cur, x_orig, target, subs)

    def generate_population(self, x_orig, nghbr_list,
                            probs, target, pop_size):
        
        return np.array(
            [self.perturb(
                x_orig, x_orig, nghbr_list,
                nghbr_dist, probs, target
             )
             for _ in range(pop_size)]
        )

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def attack(self, x_orig, target, sentence=None, weights=None,
               n_candidates=10, n_substitutes=10):
        x_adv = x_orig.copy()
        
        nghbr_list = self.LS.get_candidates(words=x_orig,
                                            n_candidates=n_candidates,
                                            n_substitutes=n_substitutes,
                                            sentence=sentence)
        nghbr_len = [len(list_i) for list_i in nghbr_list]
        if weights is None:
            sub_probs = nghbr_len / np.sum(nghbr_len)
        else:
            sub_probs = prob_normalize(nghbr_len * weights)

        if self.filter_spec:
            for i, word in enumerate(x_orig):
                if word in self.LS.spec_words:
                    sub_probs[i] = 0.
            sub_probs = prob_normalize(sub_probs)

        pop = self.generate_population(
            x_orig, nghbr_list, sub_probs, target, self.pop_size)
        
        for i in range(self.max_iters):
            batch = self.prepare_batch(pop)
            pop_preds = self.to_numpy(self.model.predict_proba(batch))
            pop_scores = pop_preds[:, target]
            if not self.targeted:
                pop_scores = 1. - pop_scores
            
            top_attack = np.argmax(pop_scores)
            select_probs = softmax(pop_scores)

            print(f'Iter = {i} -- Best score = {np.max(pop_scores}})

            if self.targeted:
                if np.argmax(pop_preds[top_attack, :]) == target:
                    return pop[top_attack]
            else:
                if np.argmax(pop_preds[top_attack, :]) != target:
                    return pop[top_attack]
            
            elite = [pop[top_attack]]
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)

            children = [self.crossover(pop[parent1_idx[i]],
                                       pop[parent2_idx[i]])
                       for i in range(self.pop_size-1)]
            children = [self.perturb(
                        x, x_orig, nghbr_list, sub_probs, target)
                        for x in children]
            pop = np.concatenate([elite, children])

        return pop[0] if top_attack is None else pop[top_attack]