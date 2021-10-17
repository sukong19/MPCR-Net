import open3d as o3d

VOXEL_SIZE = 0.1
VISUALIZE = True


template1 = o3d.io.read_point_cloud('E:\PointNetLK\ModelNet40/tryH/test/first.xyz')


template = template1.voxel_down_sample(voxel_size=VOXEL_SIZE)	

o3d.io.write_point_cloud("template.xyz",template)
