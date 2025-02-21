# learning3d-
点云分类分割配准深度学习库
问题启发式，关注数据输入输出

# 数据加载


```python
def load_data(train, use_normals):
    if train: partition = 'train' # 这里可以做一下优化，直接两分类，不用额外参数判断
    else: partition = 'test'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #（类似的python特殊变量还有那些,如何使用）
    DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
    # os.pardir 的值是 ..（表示上一级目录），类似的os操作
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
    # glob.glob其他功能和搜索方法，[] ?? *
    # `%s` 是一个占位符，会被 `partition` 的值替换。
    # `*` 是通配符，表示匹配任意字符
       f = h5py.File(h5_name)
       if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
       # normals这里指法线
       # concatenate函数使用，矩阵拼接理解，前后维度，在哪个维度操作，哪个维度就增加，其他维度必须相等且操作后不变
       else: data = f['data'][:].astype('float32')
       label = f['label'][:].astype('int64')
       f.close()
       all_data.append(data)
       all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    # np.concatenate函数本身输入就是列表or元组
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
```

```
# 只需要关注load_data，上面已经研究

class ModelNet40Data(Dataset):
    def __init__(
       self,
       train=True,
       num_points=1024,
       download=False,
       root_dir='./',
       randomize_data=False,
       use_normals=False
    ):
       super(ModelNet40Data, self).__init__()
       self.root_dir = root_dir
       if download: download_modelnet40(root_dir=root_dir) #下载失败，原因待研究
       self.data, self.labels = load_data(root_dir, train, use_normals)
       # 只需要关注load_data，上面已经研究
       if not train: self.shapes = self.read_classes_ModelNet40()
       self.num_points = num_points
       self.randomize_data = randomize_data

    def __getitem__(self, idx):
       if self.randomize_data: current_points = self.randomize(idx)
       else: current_points = self.data[idx].copy()

       current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
       label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
	# 将标签转换为长整型（通常用于分类问题）。

       return current_points, label

    def __len__(self):
       return self.data.shape[0]

    def randomize(self, idx):
       pt_idxs = np.arange(0, self.num_points)
       np.random.shuffle(pt_idxs)
       return self.data[idx, pt_idxs].copy()
       # 这里的randomize我没看懂，打乱的索引代表所有点的索引，但第二维的索引不是xyz的坐标索引吗，
       # 而且打乱点的索引，标签的索引没有对应打乱

    def get_shape(self, label):
       return self.shapes[label]

    def read_classes_ModelNet40(self):
       DATA_DIR = os.path.join(self.root_dir, 'data')
       file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
       shape_names = file.read()
       shape_names = np.array(shape_names.split('\n')[:-1])
       return shape_names
```

```
# 只需要关注data_class，在这里也是ModelNet40Data(),上面已经研究


class RegistrationData(Dataset):
	def __init__(self, algorithm, data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise=False, additional_params={}):
		super(RegistrationData, self).__init__()
		available_algorithms = ['PCRNet', 'PointNetLK', 'DCP', 'PRNet', 'iPCRNet', 'RPMNet', 'DeepGMR']
		if algorithm in available_algorithms: self.algorithm = algorithm
		else: raise Exception("Algorithm not available for registration.")
		
		self.set_class(data_class)
		self.partial_template = partial_template
		# partial == 裁剪
		self.partial_source = partial_source
		self.noise = noise
		self.additional_params = additional_params
		# 包含其他配置参数，例如邻近点的数量、是否使用掩膜等。
		self.use_rri = False

		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
			from .. ops.transform_functions import PCRNetTransform
			self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)
		if self.algorithm == 'PointNetLK':
			from .. ops.transform_functions import PNLKTransform
			self.transforms = PNLKTransform(0.8, True)
		if self.algorithm == 'RPMNet':
			from .. ops.transform_functions import RPMNetTransform
			self.transforms = RPMNetTransform(0.8, True)
		if self.algorithm == 'DCP' or self.algorithm == 'PRNet':
			from .. ops.transform_functions import DCPTransform
			self.transforms = DCPTransform(angle_range=45, translation_range=1)
		if self.algorithm == 'DeepGMR':
			self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
			from .. ops.transform_functions import DeepGMRTransform
			self.transforms = DeepGMRTransform(angle_range=90, translation_range=1)
			if 'nearest_neighbors' in self.additional_params.keys() and self.additional_params['nearest_neighbors'] > 0:
				self.use_rri = True
				self.nearest_neighbors = self.additional_params['nearest_neighbors']

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def __getitem__(self, index):
		template, label = self.data_class[index]
        # 只需要关注data_class，在这里也是ModelNet40Data(),上面已经研究
		self.transforms.index = index				# for fixed transformations in PCRNet.
		source = self.transforms(template)

		# Check for Partial Data.
		if self.additional_params.get('partial_point_cloud_method', None) == 'planar_crop':
			source, gt_idx_source = planar_crop(source)
			template, gt_idx_template = planar_crop(template)
			intersect_mask, intersect_x, intersect_y  = np.intersect1d(gt_idx_source, gt_idx_template, return_indices=True)

			self.template_mask = torch.zeros(template.shape[0])
			self.source_mask = torch.zeros(source.shape[0])
			self.template_mask[intersect_y]  = 1
			self.source_mask[intersect_x]  = 1
		else:
			if self.partial_source: source, self.source_mask = farthest_subsample_points(source)
			if self.partial_template: template, self.template_mask = farthest_subsample_points(template)



		# Check for Noise in Source Data.
		if self.noise: source = jitter_pointcloud(source)

		if self.use_rri:
			template, source = template.numpy(), source.numpy()
			template = np.concatenate([template, self.get_rri(template - template.mean(axis=0), self.nearest_neighbors)], axis=1)
			source = np.concatenate([source, self.get_rri(source - source.mean(axis=0), self.nearest_neighbors)], axis=1)
			template, source = torch.tensor(template).float(), torch.tensor(source).float()

		igt = self.transforms.igt
		
		if self.additional_params.get('use_masknet', False):
			if self.partial_source and self.partial_template:
				return template, source, igt, self.template_mask, self.source_mask
			elif self.partial_source:
				return template, source, igt, self.source_mask
			elif self.partial_template:
				return template, source, igt, self.template_mask
		else:
			return template, source, igt
```
