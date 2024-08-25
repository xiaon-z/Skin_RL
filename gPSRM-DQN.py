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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
import random
from collections import Counter

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

    emb = K.layers.Dropout(0.05)(inputs)

    emb = K.layers.Dense(256, activation="relu")(emb)       # 对特征部分进行全连接层操作（256个神经元，ReLU）

    emb = K.layers.Dropout(0.05)(emb)       # 对特征部分进行Dropout操作

    action = K.layers.Dense(n_actions, activation=None)(emb)        # 对连接后的特征进行全连接层操作

    return K.Model(inputs=inputs, outputs=action)       # 创建模型，输入为inputs，输出为actions

def initialize_clinical_practice(clinical_cases_feat, clinical_cases_labels, dataset_size, n_classes, is_training,
                                 n_patients, set_distribution):
# 初始化临床实践数据集，clinical_cases_feat病例特征, clinical_cases_labels病例标签, dataset_size数据集大小, n_classes类别数量,
# is_training是否训练, n_patients患者数量, set_distribution数据集分布
    if is_training:     # 训练阶段
        _, counts = np.unique(clinical_cases_labels, return_counts=True)        # 提取类别和每种类别的样本数
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
        # 创建数据集：包含feat和labels，对数据进行洗牌并重复使用
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
        # 设置权重，将各类别数据集合并成一个整体数据集。通过weight实现样本均衡或不同类别数据的重点关注
        dataset_train = tf.data.Dataset.sample_from_datasets([akiec_set, bcc_set, bkl_set, df_set, mel_set, nv_set, vasc_set], weights=set_distribution)
        # print("整理数据集为：",dataset_train)
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
        # print("动作空间为", self.action_space)
        # 观察空间 n_classes维的连续空间，范围从负无穷到正无穷
        self.observation_space = spaces.Box(-1 * math.inf * np.ones((n_classes,)), math.inf * np.ones((n_classes,)))
        # Initialize state
        n_state, n_gt = get_next_patient(patients)
        self.state = n_state        # 患者的特征信息
        self.revised_state = self.state
        self.gt = n_gt      # 患者的诊断结果
        # Set shower length
        self.number_of_patients = 1

    def step(self, patients, n_patients, vocab, action):        # action表示动作的索引 vocab是动作词汇表

        reward_table = np.array([[18, -9, -7, -6, -10, -8, -5, -1],
                                 [-9, 20, -12, -11, -5, -13, -10, -1],
                                 [-7, -12, 12, -5, -13, -5, -6, -1],
                                 [-6, -11, -5, 14, -12, -6, -5, -1],
                                 [-10, -5, -13, -12, 22, -14, -11, -1],
                                 [-8, -13, -5, -6, -14, 10, -7, -1],
                                 [-5, -10, -6, -5, -11, -7, 16, -1],
                                 ], np.float32)

        self.revised_state = tf.one_hot(action, len(vocab))     # 将动作的索引转换成对应的one-hot编码向量

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


# 计算倾向得分
def compute_propensity_scores(features, labels):
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    lr.fit(features, labels)
    propensity_scores = lr.predict_proba(features)
    return propensity_scores
# 根据倾向得分设置采样优先级
def set_sample_priority(propensity_scores):
    # 倾向得分越接近1，优先级越低
    probs = 1 / (propensity_scores + 1e-5)      # 避免除零错误
    probs = probs/np.sum(probs)
    return probs

def main(_):
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 0.10  # Epsilon greedy parameter
    epsilon_min = 0.05  # Minimum epsilon greedy parameter
    epsilon_max = 0.15  # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken

    #### Import Datasets ####
    tf1.enable_eager_execution()        # 启动TensorFlow1.x

    database = pd.read_csv('data/vectorDB.csv')     # 读取数据集

    database.head()     # 显示数据集的前几行

    labels = np.asarray(database['dx'])     # 提取数据集中名为“dx”的列，转换成Numpy数组存储在labels

    labels[labels == 'scc'] = 'akiec'       # 将标签数据中值为‘scc’的元素替换成‘akiec’
    le = preprocessing.LabelEncoder()       # 创建LabelEncoder对象，为将标签编码为整数
    le.fit(labels)      # 对标签数据拟合

    vocab = le.classes_     # 获取le对象中的类别列表，及词汇表
    # print("词汇表为：",vocab)
    n_words = len(vocab)        # 计算词汇表不同类别的数量
    # print("词汇表不同类别数量：",n_words)

    if Flags.use_unknown:       # 如果使用了未知标记，则添加‘unkn’到词汇表中
                vocab = np.append(vocab, 'unkn')

    features1 = np.load("data/nmed_rn34_ham10k_vectors.npy")        # 加载特征数据

    features = features1
    print("特征维度：", features.shape)

    _, counts = np.unique(labels, return_counts=True)

    # 数据集采样权重设计
    # counts = counts / np.sum(counts)
    total = counts.sum()
    # inverse_frequence = total / counts
    # log_inverse_frequence = np.log(inverse_frequence)
    # sum_log = sum(log_inverse_frequence)
    # counts = log_inverse_frequence / sum_log
    frequence = np.log(total)/np.log(counts)
    counts = frequence/frequence.sum()
    # s = counts[4] +counts[5]
    # counts[4]=s/2
    # counts[5]=s/2
    # counts = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    print("采样集权重为：", counts)
    labels_cat = le.transform(labels)       # 转换标签为整数编码形式，不会改变labels

    train_feat, val_feat, train_labels, val_labels = train_test_split(features, labels_cat, test_size=0.20,
                                                                      random_state=111, stratify=labels_cat)
    # 统计验证集中每个类别样本数
    _, count_train = np.unique(train_labels, return_counts=True)
    _, count_val = np.unique(val_labels, return_counts=True)
    print("训练集样本数", count_train)
    print("验证集样本数", count_val)
    # 初始化临床实践对象
    patients = initialize_clinical_practice(train_feat, train_labels, train_labels.shape[0], True, n_words,
                                            Flags.n_patients, counts)
    # 创建皮肤科医生对象
    derm = Dermatologist(patients, n_words, vocab)
    # 创建Q网络
    q_network = create_q_model(derm.state.shape[0], len(vocab))
    # 打印Q网络模型的摘要信息(层名、输出形状、参数数量等信息）
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
    probs_history = []      # 经验池初始采样优先级
    true_history = []       # 存放经验池中样本的真实类别
    episode_reward_history = []
    episode_val_reward_history = []
    validation_bacc_history = []
    mel_history = []
    unk_history = []
    best_bacc = 0
    best_reward = -1 * math.inf
    iter_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 20
    # Number of frames for exploration
    epsilon_greedy_frames = 100000.0        # ε-greedy策略中 ε 的值100000
    max_memory_length = 10000       # 经验回放中存储的最大记忆长度
    # Train the model after 4 actions表示在进行多少次动作后开始训练模型
    update_after_actions = 10
    update_target_network = train_feat.shape[0]     # 更新目标网络的频率
    # update_target_network = 20
    loss_function = K.losses.Huber()         # 使用 Huber 损失函数作为训练模型时的损失函数

    new_reward = [[0] * 7 for i in range(7)]
    num_reward = [[0] * 7 for i in range(7)]

    for episode in range(Flags.n_episodes):
        i = 1
        print('Starting episode ', episode)
        done = False
        episode_score = 0
        episode_val_score = 0
        state = derm.state
        gt_r = derm.gt
        n_not_random = 0
        while not done:
            try:
                iter_count += 1     # 迭代次数计数器
                # 前20步采取完全随机的动作，后面根据epsilon-greedy选动作
                if iter_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    action = derm.action_space.sample()
                else:
                    state_tensor = tf.convert_to_tensor(state)      # 将当前状态 state 转换为张量
                    state_tensor = tf.expand_dims(state_tensor, 0)     # 在张量的第一维上增加一个维度，用于适配神经网络的输入形状
                    action_probs = q_network(state_tensor, training=False)      # 预测当前状态下各个动作的概率
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()     # 概率最大的动作索引

                    n_not_random += 1       # 记录非随机选择动作的次数

                # 更新并限制epsilon的数值
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                revised_state, n_state, reward, done, gt = derm.step(patients, Flags.n_patients, vocab, action)

                episode_score += reward
                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(n_state)
                done_history.append(done)
                rewards_history.append(reward)
                probs_history.append(0.5)
                true_history.append(gt)
                state = n_state
                i += 1
                _, sample_counter = np.unique(action_history, return_counts=True)
                min_count = np.min(sample_counter)
                # Update every fourth frame 更新Q网络参数
                if iter_count % update_after_actions == 0 and len(done_history) > 100 and min_count >= 10:
                    # Get indices of samples for replay buffers
                    # 均匀采样
                    # indices = np.random.choice(range(len(done_history)), size=100)      # 从done_history列表的索引范围中随机选择100个索引
                    # 优先级采样
                    propensity_scores = compute_propensity_scores(state_history, action_history)
                    probs = set_sample_priority(propensity_scores)

                    for i, p in enumerate(probs):
                        j = true_history[i]
                        if p[j] <= 0.5:
                            probs_history[i] = probs_history[i]+p[j]
                        else:
                            probs_history[i] = 1-p[j]

                    index_action =[(index, action)for index, action in enumerate(action_history)]     # 存放动作的索引和值的元组
                    vocab_cat = le.transform(vocab)
                    action_indices = {action:[] for action in vocab_cat}        # 某动作的所有样本索引list
                    probs_indices = {action:[] for action in vocab_cat}
                    for i, a in index_action:
                        if a in action_indices:
                            action_indices[a].append(i)
                            probs_indices[a].append(probs_history[i])
                    # if episode < 10:
                    #     print("分类的优先级：",probs_indices)
                    # 归一化
                    normalized_probs_indices = []
                    for i in np.arange(len(probs_indices)):
                        # print("行prob",probs_indices[i])
                        prob = np.array(probs_indices[i])
                        total = np.sum(prob)
                        if total == 0:
                            total = 1e-10
                        normalized_probs = [pb/total for pb in prob]
                        normalized_probs_indices.append(normalized_probs)
                    probs_indices = normalized_probs_indices
                    s_indices = {}
                    for a,indices in action_indices.items():
                        # 倾向得分从经验池采样
                        s_indices[a] = np.random.choice(action_indices[a], replace=False, size=min_count, p=probs_indices[a])
                        state_sample = np.array([state_history[i] for i in s_indices[a]])
                        state_next_sample = np.array([state_next_history[i] for i in s_indices[a]])
                        rewards_sample = [rewards_history[i] for i in s_indices[a]]
                        action_sample = [action_history[i] for i in s_indices[a]]
                        done_sample = tf.convert_to_tensor([float(done_history[i]) for i in s_indices[a]])
                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                        future_rewards = target_network.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                        updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                    # 如果是最后状态，将更新后的Q值设置为-1
                        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # 创建一个one-hot编码的掩码，以便只计算更新后的Q值对应动作的损失值
                        masks = tf.one_hot(action_sample, len(vocab))

                        with tf.GradientTape() as tape:     # 创建上下文管理器，用于记录计算梯度的过程
                        # 预测Q值
                            q_values = q_network(state_sample, training=True)
                        # 得到实际采取动作后的Q值
                            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # 计算损失
                            loss = loss_function(updated_q_values, q_action)

                    # 应用梯度更新神经网络
                        grads = tape.gradient(loss, q_network.trainable_variables)
                        optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
                # 隔一定步数更新目标网络参数为当前网络的参数
                if iter_count % update_target_network == 0:
                    # update the the target network with new weights
                    target_network.set_weights(q_network.get_weights())
                    # print("已更新",iter_count)

                # 限制经验池中数量
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]


            except tf.python.framework.errors_impl.OutOfRangeError:
                done = True
                break

        # print('The episode duration was ', i - 1)
        # print('The episode reward was ', episode_score)
        # print('The number of not random actions was ', n_not_random)


        # Update running reward to check condition for solving
        episode_reward_history.append(episode_score)

        ## 验证组 ##
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

                state_tensor = tf.convert_to_tensor(state)      # 转换为Tensor对象
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = q_network(state_tensor, training=False)      # 用Q网络预测动作概率
                # 选概率最高的动作
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

        # print('The reward of the validation episode was ', episode_val_score)
        # 平衡准确率：每个类别准确率的平均值
        # print('The balanced accuracy was ', metrics.balanced_accuracy_score(true_label, error))     # 计算真实标签和预测出标签的平衡准确度
        # print("预测结果是：",error)
        # print("预测结果维度：",error.shape)
        episode_val_reward_history.append(episode_val_score)        # 每个周期验证奖励分数
        validation_bacc_history.append(metrics.balanced_accuracy_score(true_label, error))      # 记录验证阶段的平衡准确率历史
        mel_history.append(mel_count)       # 每个周期中的MEL标签计数
        unk_history.append(unk_count)       # 将每个周期中的未知标签计数
        # 计算当前的平衡准确率，并与之前记录的最佳平衡准确率进行比较
        if best_bacc < metrics.balanced_accuracy_score(true_label, error):
            history_report_bacc = metrics.classification_report(true_label, error, digits=3)        # 生成真实标签和错误标签之间的分类报告，包括精确度、召回率、F1值等指标，保留3位小数
            history_cov_bacc = metrics.confusion_matrix(true_label, error)      # 计算真实标签和错误标签之间的混淆矩阵
            best_bacc = metrics.balanced_accuracy_score(true_label, error)
            q_network.save_weights('models/best_q_network_diagnosis_bacc', save_format='tf') # 保存当前表现最佳的Q网络的权重到指定路径

            roc_auc = [0]*n_words
            fpr = [0]*n_words
            tpr = [0]*n_words
            # print(f"tlabel维度:{true_label.shape},pres的维度：{error.shape}")
            for i in range(n_words):
                # 正类是第i类，负类是其他类
                y_true = (true_label == i).astype(int)
                y_pred = (error == i).astype(int)
                # 检查是否存在正例和负例
                if not np.any(y_true) or not np.any(y_pred):
                    continue
                # 计算FPR和TPR
                fpr[i], tpr[i], _ = roc_curve(y_true, y_pred)
                # print(f"fpr为{fpr[i]}tpr为{tpr[i]}")
                # 计算AUC
                roc_auc[i] = auc(fpr[i], tpr[i])
                # print(f"auc面积为：{roc_auc[i]}")

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

        # 值函数统计
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
                # _, state, reward, done, _ = derm.step(patients_val, val_labels.shape[0], vocab, action)

                for j in range(7):
                    new_reward[derm.gt][j] += action_probs[0][j].numpy()
                    # new_reward[derm.gt][j] += reward
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
            new_reward[i][j] = new_reward[i][j] - min + 1
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
    # plt.plot([0, 1],[0, 1], color='navy', lw=2, linestyle='--')
    # print(f"tpr是{tprs},fpr是{fprs}")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.facecolor'] = 'white'
    for i in range(n_words):
        plt.plot(fpr[i], tpr[i], label=f'ROC曲线（类别{i})(AUC = {roc_auc[i]:.2f})')
    plt.legend(fontsize=40)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    # plt.grid(False)
    plt.show()
    # 绘制平均ROC曲线
    plt.figure(1)
    mean_fpr = np.mean(fpr, axis=0)
    mean_tpr = np.mean(tpr, axis=0)
    print(f"平均fpr{mean_fpr},平均tpr{mean_tpr}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(mean_fpr, mean_tpr, label=f'平均ROC曲线(AUC = {np.mean(roc_auc):.2f})')
    plt.legend(fontsize=40)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average eceiver Operating Characteristic')
    plt.legend(loc='lower right')
    # plt.grid(False)
    plt.show()

    plt.figure(2)
    plt.plot(episode_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Train')
    # plt.grid(False)
    plt.show()

    plt.figure(3)
    plt.plot(episode_val_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Val')
    # plt.grid(False)
    plt.show()

    plt.figure(4)
    plt.plot(validation_bacc_history)
    plt.xlabel('Episodes')
    plt.ylabel('RL BAcc')
    # plt.grid(False)
    plt.show()

    plt.figure(5)
    plt.plot(mel_history)
    plt.xlabel('Episodes')
    plt.ylabel('Number Melanoma Decisions')
    # plt.grid(False)
    plt.show()

    plt.figure(6)
    plt.plot(unk_history)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Unknown Decisions')
    # plt.grid(False)
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
        default=160,
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
