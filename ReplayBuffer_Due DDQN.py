import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn import preprocessing


# 计算倾向得分
def compute_propensity_scores(features, labels):
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    lr.fit(features, np.array(labels).reshape(-1,))

    propensity_scores = lr.predict_proba(features)

    return propensity_scores


# 根据倾向得分设置采样优先级
def set_sample_priority(propensity_scores):
    # 倾向得分越接近1，优先级越低
    probs = 1 / (propensity_scores + 1e-5)  # 避免除零错误
    probs = probs / np.sum(probs)
    return probs


class ReplayBuffer(object):
    def __init__(self, max_size, vocab_cat):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        # self.vocab = vocab
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        # self.done_history = []
        self.probs_history = []  # 经验池初始采样优先级
        self.true_history = []  # 存放经验池中样本的真实类别
        self.vocab_cat = vocab_cat

        self.action_sample_act = []
        self.state_sample_act = []
        self.state_next_sample_act = []
        self.rewards_sample_act = []
        # self.done_sample_act = []
        self.true_sample_act = []


    def add(self, state, action, reward, next_state, gt):
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(next_state)
        # self.done_history.append(done)
        self.rewards_history.append(reward)
        self.probs_history.append(0.5)
        self.true_history.append(gt)
        self.size += 1

        _, sample_counter = np.unique(self.action_history, return_counts=True)
        min_count = np.min(sample_counter)
        if (len(self.action_history) != len(self.state_history) or len(self.state_history) != len(self.state_next_history) or
                len(self.rewards_history) != len(self.probs_history) or len(self.probs_history) != len(self.true_history)):
            # 输出长度
            print("Lengths of lists are not equal:")
            print(f"action_history: {len(self.action_history)}")
            print(f"state_history: {len(self.state_history)}")
            print(f"state_next_history: {len(self.state_next_history)}")
            # print(f"done_history: {len(self.done_history)}")
            print(f"rewards_history: {len(self.rewards_history)}")
            print(f"probs_history: {len(self.probs_history)}")
            print(f"true_history: {len(self.true_history)}")
            # 终止程序
            raise ValueError("Lengths of lists are not equal. Terminating the program.")
        if self.size > self.max_size:          # 限制replaybuffer中经验个数不超过max_size
            del self.rewards_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.action_history[:1]
            # del self.done_history[:1]
            del self.probs_history[:1]
            del self.true_history[:1]
            # del self.probs[:1]
            self.size -= 1

        return min_count

    def sample(self, batch):
        state_all = []
        action_all = []
        reward_all = []
        next_state_all = []
        # done_all = []
        true_all = []

        # 计算倾向得分

        propensity_scores = compute_propensity_scores(self.state_history, self.action_history)

        probs = set_sample_priority(propensity_scores)          # (1000,7)
        # print("状态：",np.array(state_history).shape)
        # print("倾向得分：", probs.shape)

        for i, p in enumerate(probs):
            j = self.true_history[i]
            if p[j] <= 0.5:         # 预测成功的概率很小，提高该样本被采样的机会
                self.probs_history[i] = self.probs_history[i] + p[j]
            else:                   # 预测成功的概率较大，减小被采样的机会
                self.probs_history[i] = 1 - p[j]

        probs_history = np.squeeze(self.probs_history)
        action_history = np.squeeze(self.action_history)
        # state_history = np.squeeze(self.state_history)



        # print(np.shape(probs_history))
        # print("动作", np.shape(action_history))
        # print("状态", np.shape(state_history))
        # index_action = [(index, action) for index, action in enumerate(self.action_history)]
        # print("索引动作", index_action)
        probs_history /= probs_history.sum()
        s_indices = np.random.choice(len(action_history), replace=False, size=batch, p=probs_history)     # 每个类别根据倾向得分随机采样counts条样本下标
        # print("index", s_indices)
        state_sample = np.array([self.state_history[i] for i in s_indices])
        # print("sate_sample: ", state_sample)
        state_next_sample = np.array([self.state_next_history[i] for i in s_indices])
        rewards_sample = np.array([self.rewards_history[i] for i in s_indices])
        action_sample = np.array([self.action_history[i] for i in s_indices])
        # true_sample = tf.convert_to_tensor([float(self.true_history[i]) for i in s_indices])
        true_sample = np.array([self.true_history[i] for i in s_indices])
        # print("state",state_sample)
        # action_indices = {action: [] for action in self.vocab_cat}  # 某动作的所有样本索引list
        # probs_indices = {action: [] for action in self.vocab_cat}
        # print("动作索引对", action_indices)
        # print("youxianji索引对", probs_indices)

        # for i, a in index_action:
        #
        #     if a in action_indices:
        #         action_indices[a].append(i)         # 记录每个类别的样本下标
        #         probs_indices[a].append(self.probs_history[i])          # 记录每个类别的倾向得分
        #     # 归一化
        # normalized_probs_indices = []
        # for i in np.arange(len(probs_indices)):
        #     # print("行prob",probs_indices[i])
        #     prob = np.array(probs_indices[i])
        #     total = np.sum(prob)
        #     if total == 0:
        #         total = 1e-10
        #     normalized_probs = [pb / total for pb in prob]
        #     normalized_probs_indices.append(normalized_probs)
        # probs_indices = normalized_probs_indices
        # counts = batch // 7
        # s_indices = {}
        # for a, indices in action_indices.items():
        #     # 倾向得分从经验池采样
        #     s_indices[a] = np.random.choice(action_indices[a], replace=False, size=counts,
        #                                     p=probs_indices[a])     # 每个类别根据倾向得分随机采样counts条样本下标
        #     # print("采样下标",s_indices[a])
        #     state_sample = np.array([self.state_history[i] for i in s_indices[a]])
        #     state_next_sample = np.array([self.state_next_history[i] for i in s_indices[a]])
        #     rewards_sample = np.array([self.rewards_history[i] for i in s_indices[a]])
        #     action_sample = np.array([self.action_history[i] for i in s_indices[a]])
        #     # done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in s_indices[a]])
        #     true_sample = tf.convert_to_tensor([float(self.true_history[i]) for i in s_indices[a]])
        #     state_all.extend([item for item in state_sample])
        #     next_state_all.extend([item for item in state_next_sample])
        #     reward_all.extend([item for item in rewards_sample])
        #     action_all.extend([item for item in action_sample])
        #     # done_all.extend([item for item in done_sample])
        #     true_all.extend([item for item in true_sample])
            # print("存储的action",action_all)
            # done_sample = np.array([float(self.done_history[i]) for i in s_indices[a]])
            # true_sample = np.array([float(self.done_history[i]) for i in s_indices[a]])

        return (
            state_sample,
            action_sample,
            rewards_sample,
            state_next_sample,
            true_sample

        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state_history)
        np.save(f"{save_folder}_action.npy", self.action_history)
        np.save(f"{save_folder}_next_state.npy", self.state_next_history)
        np.save(f"{save_folder}_reward.npy", self.rewards_history)
        # np.save(f"{save_folder}_done.npy", self.done_history)
        np.save(f"{save_folder}_true.npy", self.true_history)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        # self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]