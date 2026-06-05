import scipy.io as sio
import numpy as np
import hdf5storage as hdf5
#

train_ratio = 1
patch = 1

def split_and_sample(input_array, sample_percentage=5):
    unique_values = np.unique(input_array)# 0, 1, 2, 3
    unique_values = unique_values[unique_values != 0]

    train_array = np.zeros_like(input_array)
    test_array = np.zeros_like(input_array)
    val_array = np.zeros_like(input_array)
    np.random.seed(1)
    for value in unique_values:
        mask = (input_array == value)
        value_positions = np.where(mask)

        # 随机选择5%的位置
        num_samples = int(np.ceil(np.sum(mask) * sample_percentage / 100))
        sampled_positions = np.random.choice(np.arange(len(value_positions[0])), size=num_samples, replace=False)

        # 将选中的位置的值放入train_array，其余放入test_array
        train_positions = (value_positions[0][sampled_positions], value_positions[1][sampled_positions])
        test_positions = (np.delete(value_positions[0], sampled_positions), np.delete(value_positions[1], sampled_positions))
        #验证集从测试集中随机选1%
        num_samples_val = int(np.ceil(np.sum(mask) * 1 / 100))
        sampled_val_positions = np.random.choice(np.arange(len(test_positions[0])), size=num_samples_val, replace=False)
        val_posotions = (test_positions[0][sampled_val_positions], test_positions[1][sampled_val_positions])
        test_positions = (np.delete(test_positions[0], sampled_val_positions), np.delete(test_positions[1], sampled_val_positions))

        train_array[train_positions] = input_array[train_positions]
        test_array[test_positions] = input_array[test_positions]
        val_array[val_posotions] = input_array[val_posotions]

    return train_array, test_array, val_array

def generate_TR_TE(data_name,ratio, j):#data_name:"512_512",ratio:1,5
    train_ratio = ratio
    # label = hdf5.loadmat(f'{data_name}/{data_name}_label.mat')['label']#label  aa
    label = hdf5.loadmat(f'F:/pycharm/PyCharm_projects/postGraduate/sky/cnn_cla/1300_1200/pattch/label{j}.mat')['label']
    print(label.shape)
    print(label.dtype)
    print(type(label))
    # feature = hdf5.loadmat('./data_name/data_name_9.mat')['feature']
    train_result, test_result, val_result = split_and_sample(label,train_ratio)
    # print(train_result.shape, test_result.shape, val_result.shape)
    #保存训练集，测试集
    #sio.savemat('./data_name/data_name_%s_split.mat' %train_ratio, {'TR': train_result,'TE': test_result})
    return train_result, test_result, val_result, label
    # sio.savemat('./512_512/512_512_%s_split.mat' %train_ratio, {'TR': train_result,'TE': test_result, 'feature': feature})
   # sio.savemat('./%s/T%d_%s_split.mat'%(dataset,patch,train_ratio), {'TR': train_result,'TE': test_result, 'feature': feature})
    '''
    以下是将分好的结果输出展示
    调用，只需要返回上边分好的训练集和测试集
    '''
   #  all_data_set = [train_result,test_result]
   #  for data_set in all_data_set:
   #      unique_elements, counts = np.unique(data_set, return_counts=True)
   #      #写入文件
   #      result_array = np.column_stack((unique_elements, counts))
   #      # 将结果保存到文本文件
   #      # with open('./512_512/512_512_%s_split.txt'%train_ratio, 'a+') as file:
   #      with open('./%s/label%d_%s_split.txt'%(dataset,patch,train_ratio), 'a+') as file:
   #          file.write('*************\n')
   #          #np.savetxt(file, result_array, fmt='%d', delimiter='\t')
   #      # 将唯一元素和计数打印出来
   #      print(f'train_ratio:{i} , patch:{j}')
   #      for element, count in zip(unique_elements, counts):
   #          print(f"{element} ：{count}")
   #      print('********************************')
