# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:13:09 2018

@author: huahua
"""

import os
import numpy as np
import tensorflow as tf
from input_data import OCRIter
import model
#from genplate import *
import time
import datetime

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_w = 272
img_h = 72
num_label=7
batch_size = 8
count =30000
learning_rate = 0.0001

#默认参数[N,H,W,C]
image_holder = tf.placeholder(tf.float32,[batch_size,img_h,img_w,3])
label_holder = tf.placeholder(tf.int32,[batch_size,7])
keep_prob = tf.placeholder(tf.float32)

logs_train_dir = './home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/train_logs_50000/'

def get_batch():
    data_batch = OCRIter(batch_size,img_h,img_w)
    image_batch,label_batch = data_batch.iter()

    image_batch1 = np.array(image_batch)
    label_batch1 = np.array(label_batch)

    return image_batch1,label_batch1


train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7= model.inference(image_holder,keep_prob)

train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7 = model.losses(train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7,label_holder) 
train_op1,train_op2,train_op3,train_op4,train_op5,train_op6,train_op7 = model.trainning(train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7,learning_rate)

train_acc = model.evaluation(train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7,label_holder)

input_image=tf.summary.image('input',image_holder)
#tf.summary.histogram('label',label_holder) #label的histogram,测试训练代码时用，参考:http://geek.csdn.net/news/detail/197155

summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))  #运行日志
sess = tf.Session() 

train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

start_time1 = time.time()    

for step in range(count): 

    x_batch,y_batch = get_batch()

    start_time2 = time.time()   
    time_str = datetime.datetime.now().isoformat()

    feed_dict = {image_holder:x_batch,label_holder:y_batch,keep_prob:0.5}
    _,_,_,_,_,_,_,tra_loss1,tra_loss2,tra_loss3,tra_loss4,tra_loss5,tra_loss6,tra_loss7,acc,summary_str= sess.run([train_op1,train_op2,train_op3,train_op4,train_op5,train_op6,train_op7,train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7,train_acc,summary_op],feed_dict)
    train_writer.add_summary(summary_str,step)
    duration = time.time()-start_time2
    tra_all_loss =tra_loss1+tra_loss2+tra_loss3+tra_loss4+tra_loss5+tra_loss6+tra_loss7

    #print(y_batch)  #仅测试代码训练实际样本与标签是否一致

    if step % 10== 0:
        sec_per_batch = float(duration)
        print('%s : Step %d,train_loss = %.2f,acc= %.2f,sec/batch=%.3f' %(time_str,step,tra_all_loss,acc,sec_per_batch))

    if step%2000==0 or (step+1)==count:
        checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
        saver = tf.train.Saver()
        saver.save(sess,checkpoint_path,global_step=step)
sess.close()       
print(time.time()-start_time1)
