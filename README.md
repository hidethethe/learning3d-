# learning3d-
点云分类分割配准深度学习库
问题启发式，关注数据输入输出
首先主观评价，代码较为繁杂，可读性一般。
# 数据加载
'''python
def load_data(train, use_normals):
    if train: partition = 'train' # 这里可以做一下优化，直接两分类，不用额外参数判断
    else: partition = 'test'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) ####（类似的特殊变量还有那些）
    DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')  ####os.pardir 的值是 ..（表示上一级目录）
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)): #### glob.glob其他功能和搜索方法，[] ?? *
       f = h5py.File(h5_name)
       if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32') #### concatenate
       else: data = f['data'][:].astype('float32')
       label = f['label'][:].astype('int64')
       f.close()
       all_data.append(data)
       all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
'''
