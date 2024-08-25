from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import math
import sklearn.metrics as metrics
from tqdm import tqdm
from gym import Env, spaces
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim

from tensorflow.python.keras import backend
from tensorflow.keras.backend import clear_session

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.keras as K

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

from sklearn.feature_extraction.text import CountVectorizer


def read_and_decode(dataset, batch_size, is_training, data_size, n_patients):
    if is_training:     # 处于训练阶段
        dataset = dataset.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)     # 打乱数据集顺序
        dataset = dataset.batch(batch_size, drop_remainder=False)       # 进行分批处理，并重复使用数据集中的数据
        dataset = dataset.repeat(None)      # 无限重复
    else:
        dataset = dataset.prefetch(buffer_size=data_size // batch_size)     # 对数据进行预取
        dataset = dataset.batch(batch_size, drop_remainder=False)       # 进行分批处理，并重复使用数据集中的数据
        dataset = dataset.repeat(None)      # 无限重复
    return dataset

def create_q_model(n_classes, n_actions):
    inputs = K.layers.Input(n_classes)      # 输入层

    feat = K.layers.Lambda(lambda x: x[:, 0:n_classes - n_actions + 1])(inputs)     # 截取输入的部分特征

    emb = K.layers.Dropout(0.05)(feat)      # 丢弃部分神经元，防止过拟合

    prob = K.layers.Lambda(lambda x: x[:, n_classes - n_actions + 1:n_classes])(inputs)     # 截取输入的概率部分

    emb = K.layers.Dense(256, activation="relu")(emb)       # 对特征部分进行全连接层操作（256个神经元，ReLU）

    emb = K.layers.Dropout(0.05)(emb)       # 对特征部分进行Dropout操作

    emb = K.layers.Concatenate(axis=1)([emb, prob])

    value_stream = K.layers.Dense(1)(emb)     # 值函数分支

    advantage_stream = K.layers.Dense(n_actions)(emb)     # 优势函数分支

    q_values = value_stream + (advantage_stream - tf.reduce_mean(advantage_stream, axis=1, keepdims=True))

    return tf.keras.Model(inputs=inputs, outputs=q_values)


def initialize_clinical_practice(clinical_cases_feat, clinical_cases_labels, dataset_size, n_classes, is_training,
                                 n_patients, set_distribution):
# 初始化临床实践数据集，clinical_cases_feat病例特征, clinical_cases_labels病例标签, dataset_size数据集大小, n_classes类别数量,
# is_training是否训练, n_patients患者数量, set_distribution数据集分布
    if is_training:     # 训练阶段
        _, counts = np.unique(clinical_cases_labels, return_counts=True)        # 提取特征和标签
        # 根据不同类别分别提取特征和标签
        akiec = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 0), axis=0))
        akiec_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 0), axis=0))

        bcc = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 1), axis=0))
        bcc_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 1), axis=0))

        bkl = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 2), axis=0))
        bkl_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 2), axis=0))

        df = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 3), axis=0))
        df_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 3), axis=0))

        mel = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 4), axis=0))
        mel_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 4), axis=0))

        nv = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 5), axis=0))
        nv_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 5), axis=0))

        vasc = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 6), axis=0))
        vasc_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 6), axis=0))
        # 创建数据集
        akiec_set = tf.data.Dataset.from_tensor_slices((akiec, akiec_labels)).shuffle(buffer_size=counts[0],
                                                                                      reshuffle_each_iteration=True).repeat()
        bcc_set = tf.data.Dataset.from_tensor_slices((bcc, bcc_labels)).shuffle(buffer_size=counts[1],
                                                                                reshuffle_each_iteration=True).repeat()
        bkl_set = tf.data.Dataset.from_tensor_slices((bkl, bkl_labels)).shuffle(buffer_size=counts[2],
                                                                                reshuffle_each_iteration=True).repeat()
        df_set = tf.data.Dataset.from_tensor_slices((df, df_labels)).shuffle(buffer_size=counts[3],
                                                                             reshuffle_each_iteration=True).repeat()
        mel_set = tf.data.Dataset.from_tensor_slices((mel, mel_labels)).shuffle(buffer_size=counts[4],
                                                                                reshuffle_each_iteration=True).repeat()
        nv_set = tf.data.Dataset.from_tensor_slices((nv, nv_labels)).shuffle(buffer_size=counts[5],
                                                                             reshuffle_each_iteration=True).repeat()
        vasc_set = tf.data.Dataset.from_tensor_slices((vasc, vasc_labels)).shuffle(buffer_size=counts[6],
                                                                                   reshuffle_each_iteration=True).repeat()
        # 设置权重，将各类别数据集合并成一个整体数据集
        dataset_train = tf.data.Dataset.sample_from_datasets([akiec_set, bcc_set, bkl_set, df_set, mel_set, nv_set, vasc_set], weights=set_distribution)
        # 设定批次大小为1
        dataset_train = dataset_train.batch(1)

    else:
        dataset_train = tf.data.Dataset.from_tensor_slices((clinical_cases_feat, clinical_cases_labels))

        dataset_train = read_and_decode(dataset_train, 1, is_training, dataset_size, n_patients)
    # 返回数据集迭代器，用于迭代访问数据集
    patients = iter(dataset_train)

    return patients

def get_next_patient(patients):     # 从患者数据集中获取下一个患者的特征和诊断信息
    patient_scores, patient_diagnostics = patients.get_next()
    # 压缩患者特征，并转换为NumPy数组，取第一个元素作为诊断结果
    return np.squeeze(patient_scores), patient_diagnostics.numpy()[0]


class Dermatologist(Env):

    def __init__(self, patients, n_classes, vocab):
        # 动作空间, either skin lesion classes or don't know
        self.action_space = spaces.Discrete(len(vocab))
        # 观察空间 n_classes维的连续空间，范围从负无穷到正无穷
        self.observation_space = spaces.Box(-1 * math.inf * np.ones((n_classes,)), math.inf * np.ones((n_classes,)))
        # Initialize state
        n_state, n_gt = get_next_patient(patients)
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt
        # Set shower length
        self.number_of_patients = 1

    def step(self, patients, n_patients, vocab, action):
        reward_table = np.array([[2, -2, -3, -3, -2, -3, -3, -1],
                                 [-2, 3, -4, -4, -2, -4, -4, -1],
                                 [-2, -2, 1, -2, -3, -2, -2, -1],
                                 [-2, -2, -2, 1, -3, -2, -2, -1],
                                 [-4, -3, -5, -5, 5, -5, -5, -1],
                                 [-2, -2, -2, -2, -3, 1, -2, -1],
                                 [-2, -2, -2, -2, -3, -2, 1, -1],
                                 ], np.float32)

        self.revised_state = tf.one_hot(action, len(vocab))     # 将动作转换成独热编码

        reward = reward_table[self.gt, action]      # 根据状态和动作从奖励表获取奖励值

        n_state, n_gt = get_next_patient(patients)

        old_gt = self.gt

        self.state = n_state

        self.gt = n_gt

        self.number_of_patients += 1

        # 判断是否完成所有患者的检查
        if self.number_of_patients >= n_patients:  # or old_gt != action:
            done = 1
        else:
            done = 0

            # reward += 1000

        return self.revised_state, self.state, reward, done, old_gt

    def reset(self, clinical_cases_feat, clinical_cases_labels, n_classes, dataset_size, vocab, is_training, n_patients,
              sample_distribution):     # 重置环境的状态
        # Reset clinical practice
        patients = initialize_clinical_practice(clinical_cases_feat, clinical_cases_labels, dataset_size, n_classes,
                                                is_training, n_patients, sample_distribution)
        n_state, n_gt = get_next_patient(patients)
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt
        # Reset new practice
        self.number_of_patients = 0

        return self.state, patients


def main(_):
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 0.2  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 0.2  # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken

    #### Import Datasets ####
    tf1.enable_eager_execution()

    database = pd.read_csv('data/vectorDB.csv')

    print("数据集为", database)

    database.head()

    labels = np.asarray(database['dx'])

    print("标签为", labels)

    labels[labels == 'scc'] = 'akiec'

    le = preprocessing.LabelEncoder()

    le.fit(labels)

    vocab = le.classes_

    n_words = len(vocab)

    if Flags.use_unknown:
                vocab = np.append(vocab, 'unkn')

    features1 = np.load("data/nmed_rn34_ham10k_vectors.npy")

    features2 = pd.read_csv("data/vectorDB.csv")

    features2.pop('dx')

    features2 = np.asarray(features2, dtype='float32')

    features = np.concatenate([features1, features2], axis=1)

    _, counts = np.unique(labels, return_counts=True)

    print("采样集为：", counts)

    counts = counts / np.sum(counts)
    # total = counts.sum()
    # inverse_frequence = total / counts
    # log_inverse_frequence = np.log(inverse_frequence)
    # sum_log = sum(log_inverse_frequence)
    # counts = log_inverse_frequence / sum_log
    # s = counts[4] +counts[5]
    # counts[4]=s/2
    # counts[5]=s/2
    # counts = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    print("处理后采样集为：", counts)
    labels_cat = le.transform(labels)       # 转换标签为整数编码形式

    # print("标签编号类型为",labels_cat.dtype)
    # 划分数据集为训练集和验证集
    train_feat, val_feat, train_labels, val_labels = train_test_split(features, labels_cat, test_size=0.2,
                                                                      random_state=111, stratify=labels_cat)
    # 统计验证集中每个类别样本数
    _, count_train = np.unique(val_labels, return_counts=True)

    print("验证集样本数",count_train)
    # 初始化临床实践对象
    patients = initialize_clinical_practice(train_feat, train_labels, train_labels.shape[0], True, n_words,
                                            Flags.n_patients, counts)
    # 创建皮肤科医生对象
    derm = Dermatologist(patients, n_words, vocab)

    q_network = create_q_model(derm.state.shape[0], len(vocab))
    # 打印Q网络模型的摘要信息
    q_network.summary()
    # 创建目标网络和Q网络结构相同
    target_network = create_q_model(derm.state.shape[0], len(vocab))
    # 优化器
    optimizer = K.optimizers.Adam(learning_rate=0.025, clipnorm=1.0)

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    episode_val_reward_history = []
    validation_bacc_history = []
    mel_history = []
    unk_history = []
    cnn_history = []
    best_bacc = 0
    best_reward = -1 * math.inf
    iter_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 20
    # Number of frames for exploration
    epsilon_greedy_frames = 100000.0
    # Maximum replay length
    max_memory_length = 10000
    # Train the model after 4 actions
    update_after_actions = 10
    # How often to update the target network
    update_target_network = train_feat.shape[0]
    # Using huber loss for stability
    loss_function = K.losses.Huber()

    new_reward = [[0] * 7 for i in range(7)]
    num_reward = [[0] * 7 for i in range(7)]

    # acc_class = [0, 0, 0, 0, 0, 0, 0]
    for episode in range(Flags.n_episodes):
        i = 1
        print('Starting episode ', episode)

        done = False
        episode_score = 0

        episode_val_score = 0

        state = derm.state
        # state = np.concatenate([derm.state, [0]])
        n_not_random = 0
        while not done:
            try:
                iter_count += 1
                # 根据epsilon-greedy选动作 for exploration
                if iter_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    action = derm.action_space.sample()
                else:
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = q_network(state_tensor, training=False)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                    n_not_random += 1

                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                revised_state, n_state, reward, done, _ = derm.step(patients, Flags.n_patients, vocab, action)

                episode_score += reward

                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(n_state)
                done_history.append(done)
                rewards_history.append(reward)
                state = n_state

                i += 1
                # Update every fourth frame 更新Q网络参数
                if iter_count % update_after_actions == 0 and len(done_history) > 100:
                    # Get indices of samples for replay buffers
                    # 均匀采样
                    indices = np.random.choice(range(len(done_history)), size=100)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = target_network.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, len(vocab))

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = q_network(state_sample, training=True)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, q_network.trainable_variables)
                    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
                # 隔一定步数更新目标网络参数为当前网络的参数
                if iter_count % update_target_network == 0:
                    # update the the target network with new weights
                    target_network.set_weights(q_network.get_weights())

                # Limit the state and reward history
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]


            except tf.python.framework.errors_impl.OutOfRangeError:
                done = True
                break

        print('The episode duration was ', i - 1)
        print('The episode reward was ', episode_score)
        print('The number of not random actions was ', n_not_random)
        # Update running reward to check condition for solving
        episode_reward_history.append(episode_score)

        ## Validation Phase ##
        state, patients_val = derm.reset(val_feat, val_labels, val_labels.shape[0], n_words, vocab, False,
                                         Flags.n_patients, counts)

        done = False

        error = np.array([])
        CNN_error = np.array([])
        true_label = np.array([])

        mel_count = 0

        unk_count = 0

        while not done:
            try:
                true_label = np.append(true_label, derm.gt)

                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = q_network(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                error = np.append(error, action)

                if vocab[action] == 'mel':
                    mel_count += 1
                elif vocab[action] == 'unkn':
                    unk_count += 1

                _, state, reward, done, _ = derm.step(patients_val, val_labels.shape[0], vocab, action)

                episode_val_score += reward

            except tf.python.framework.errors_impl.OutOfRangeError:
                done = True
                break

        print('The reward of the validation episode was ', episode_val_score)
        print('The balanced accuracy was ', metrics.balanced_accuracy_score(true_label, error))
        episode_val_reward_history.append(episode_val_score)
        validation_bacc_history.append(metrics.balanced_accuracy_score(true_label, error))
        mel_history.append(mel_count)
        unk_history.append(unk_count)
        # 计算当前的平衡准确率，并与之前记录的最佳平衡准确率进行比较
        if best_bacc < metrics.balanced_accuracy_score(true_label, error):
            history_report_bacc = metrics.classification_report(true_label, error, digits=3)
            history_cov_bacc = metrics.confusion_matrix(true_label, error)
            best_bacc = metrics.balanced_accuracy_score(true_label, error)
            q_network.save_weights('models/best_q_network_diagnosis_bacc', save_format='tf')

            roc_auc = [0] * n_words
            fpr = [0] * n_words
            tpr = [0] * n_words
            # print(f"tlabel维度:{true_label.shape},pres的维度：{error.shape}")
            for i in range(n_words):
                # 正类是第i类，负类是其他类
                y_true = (true_label == i).astype(int)
                y_pred = (error == i).astype(int)
                # 检查是否存在正例和负例
                if not np.any(y_true) or not np.any(y_pred):
                    continue
                fpr[i], tpr[i], _ = roc_curve(y_true, y_pred)
            # print(f"fpr为{fpr[i]}tpr为{tpr[i]}")
            # 计算AUC
                roc_auc[i] = auc(fpr[i], tpr[i])


        if best_reward < episode_val_score:
            history_cov_reward = metrics.confusion_matrix(true_label, error)
            history_report_reward = metrics.classification_report(true_label, error, digits=3)
            best_reward = episode_val_score
            q_network.save_weights('models/best_q_network_diagnosis_reward', save_format='tf')

        ##Return to train
        _, patients = derm.reset(train_feat, train_labels, train_labels.shape[0], n_words, vocab, True,
                                 Flags.n_patients, counts)

    # print("每类精确度：", acc_class / episode)

    q_network.save_weights('models/q_network_diagnosis_final',
                           save_format='tf')

    #值函数统计
    state, patients_val = derm.reset(train_feat, train_labels, train_labels.shape[0], n_words, vocab, False,
                                     Flags.n_patients, counts)
    done = False
    for i in range(100):
        while not done:
            try:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = q_network(state_tensor, training=False)

                # Take best action

                action = tf.argmax(action_probs[0]).numpy()

                for j in range(7):
                    new_reward[derm.gt][j] += action_probs[0][j].numpy()
                    num_reward[derm.gt][j] += 1
                _, state, reward, done, _ = derm.step(patients_val, val_labels.shape[0], vocab, action)
            except tf.python.framework.errors_impl.OutOfRangeError:
                done = True
                break
    min = 10000
    for i in range(7):
        for j in range(7):
            new_reward[i][j] = new_reward[i][j] / num_reward[i][j]
        if new_reward[i][i] < min:
            min = new_reward[i][i]
    for i in range(7):
        for j in range(7):
            new_reward[i][j] = new_reward[i][j]-min+1
        print(new_reward[i])
    print("最小值为", min)
    print('The scores for best validation BAcc are:')
    print(history_report_bacc)
    print(history_cov_bacc)
    print('The best BAcc was ', best_bacc)



    print('The scores for best validation Reward are:')
    print(history_report_reward)
    print(history_cov_reward)
    print('The best reward was ', best_reward)


    # 打印每个类别的AUC面积
    for i in range(n_words):
        print(f"AUC面积（类别{i}):{roc_auc[i]}")
    # 打印平均AUC面积
    print(f"平均AUC面积：{np.mean(roc_auc)}")
    # 绘制ROC曲线
    plt.figure(0)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(n_words):
        plt.plot(fpr[i], tpr[i], label=f'ROC曲线（类别{i})(AUC = {roc_auc[i]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    # 绘制平均ROC曲线
    plt.figure()
    mean_fpr = np.mean(fpr, axis=0)
    mean_tpr = np.mean(tpr, axis=0)
    print(f"平均fpr{mean_fpr},平均tpr{mean_tpr}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(mean_fpr, mean_tpr, label=f'平均ROC曲线(AUC = {np.mean(roc_auc):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average eceiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(1)
    plt.plot(episode_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Train')
    plt.show()

    plt.figure(2)
    plt.plot(episode_val_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Val')
    plt.show()

    plt.figure(3)
    plt.plot(validation_bacc_history)
    plt.xlabel('Episodes')
    plt.ylabel('RL BAcc')
    plt.show()

    plt.figure(4)
    plt.plot(mel_history)
    plt.xlabel('Episodes')
    plt.ylabel('Number Melanoma Decisions')
    plt.show()

    plt.figure(5)
    plt.plot(unk_history)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Unknown Decisions')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_patients',
        type=int,
        default=100,
        help='Number of patients per episode.'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=150,
        help='Number of episodes to play'
    )
    parser.add_argument(
        '--use_unknown',
        type=bool,
        default=False,
        help='Whether to use unknown action or not distribution'
    )
    Flags, unparsed = parser.parse_known_args()
    tf1.app.run(main=main)
