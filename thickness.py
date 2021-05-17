import trimesh
from tvtk.api import tvtk
import networkx as nx
from tvtk.common import configure_port_input_data
import numpy as np


def filter_mesh(verts, faces, mask_verts):
    mask_faces = ~np.any(np.isin(faces,np.argwhere(~mask_verts)), axis=1)
    
    remap_faces = np.arange(mask_verts.shape[0])-np.cumsum(~mask_verts)
    
    r_verts = verts[mask_verts,:]
    r_faces = remap_faces[faces[mask_faces,:]]
    
    return r_verts, r_faces


def mesh_thickness(verts, faces):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,validate=False,process=False)
    g = mesh.vertex_adjacency_graph
    
    
    cc = list(nx.connected_components(g))
    if len(cc) > 1:
        ts = []
        ss = []
        for x in cc:
            # print(len(x))
            if len(x) > 100:
                mask_verts = np.zeros(verts.shape[0], dtype=np.bool)
                mask_verts[list(x)] = True
                f_verts, f_faces = filter_mesh(verts, faces, mask_verts)
                t,s = mesh_thickness(f_verts, f_faces)
                ts.append(t)
                ss.append(s)
        
        return sum(ts) / len(ts), np.sqrt(sum([x**2 for x in ss]) / len(ss))
    
    
    cs = np.zeros(mesh.vertices.shape[0])
    n = mesh.vertex_normals
    for i in range(mesh.vertices.shape[0]):
        # nb = mesh.vertex_neighbors[i]
        nb = nx.single_source_shortest_path_length(g, i, 2)
        nb = list(nb.keys())
        cs[i] = (n[nb] @ n[i]).min()
    
    #%%
    mask_verts = cs>0.75
    mask_faces = ~np.any(np.isin(faces,np.argwhere(~mask_verts)), axis=1)
    
    remap_faces = np.arange(cs.shape[0])-np.cumsum(~mask_verts)
    
    r_cs = cs[mask_verts]
    r_verts = verts[mask_verts,:]
    r_faces = remap_faces[faces[mask_faces,:]]
    
    
    #%%
    mesh = trimesh.Trimesh(vertices=r_verts, faces=r_faces,validate=False,process=False)
    cc = list(nx.connected_components(mesh.vertex_adjacency_graph))
    
    # Only maintain significant connected components
    cc = [x for x in cc if len(x) > 500]
    
    if len(cc) == 1:
        print('Running mincut')
        n = mesh.vertex_normals
        
        ind1 = (n[:,1]-r_cs).argmin()
        ind2 = (n[:,1]+r_cs).argmax()
        
        g = mesh.vertex_adjacency_graph
        
        for u,v,d in g.edges(data=True):
            w = min(r_cs[u],r_cs[v])
            if w < 0.8:
                w = 0
        
            d['capacity'] = 100**((w-0.8)*10)
        
        v,cc = nx.minimum_cut(g,ind1,ind2)
        
        print(len(cc[0]), len(cc[1]))
    
    
    if len(cc) > 2:
        # len_cc = [len(x) for x in cc]
        # print(len_cc)
        # t = sorted(len_cc)[-2]
        # cc = [x for x in cc if len(x) >= t]
        
        print('THIS WILL FAIL FOR OTHER SCANS!')
        ts = []
        ss = []

        mask_verts = np.zeros((r_verts.shape[0],),dtype=np.bool)
        for x in cc[0]:
             mask_verts[x] = True
        
        mask_faces = ~np.any(np.isin(r_faces,np.argwhere(~mask_verts)), axis=1)
        remap_faces = np.arange(r_verts.shape[0])-np.cumsum(~mask_verts)
        
        mesh1 = tvtk.PolyData(points=r_verts[mask_verts,:], polys=remap_faces[r_faces[mask_faces,:]])
        
        mask_verts = np.zeros((r_verts.shape[0],),dtype=np.bool)
        for x in cc[1]:
            mask_verts[x] = True
        
        mask_faces = ~np.any(np.isin(r_faces,np.argwhere(~mask_verts)), axis=1)
        remap_faces = np.arange(r_verts.shape[0])-np.cumsum(~mask_verts)
        
        mesh2 = tvtk.PolyData(points=r_verts[mask_verts,:], polys=remap_faces[r_faces[mask_faces,:]])
        
        dist = tvtk.DistancePolyDataFilter()
        
        configure_port_input_data(dist, 0, mesh1)
        configure_port_input_data(dist, 1, mesh2)
        dist.signed_distance = 0
        dist.compute_second_distance = 1
        dist.update()   
        
        ts.append(np.concatenate((dist.output.point_data.scalars.to_array(),dist.get_output(1).point_data.scalars.to_array())).mean())
        ss.append(np.concatenate((dist.output.point_data.scalars.to_array(),dist.get_output(1).point_data.scalars.to_array())).std())


        mask_verts = np.zeros((r_verts.shape[0],),dtype=np.bool)
        for x in cc[2]:
             mask_verts[x] = True
        
        mask_faces = ~np.any(np.isin(r_faces,np.argwhere(~mask_verts)), axis=1)
        remap_faces = np.arange(r_verts.shape[0])-np.cumsum(~mask_verts)
        
        mesh1 = tvtk.PolyData(points=r_verts[mask_verts,:], polys=remap_faces[r_faces[mask_faces,:]])
        
        mask_verts = np.zeros((r_verts.shape[0],),dtype=np.bool)
        for x in cc[3]:
            mask_verts[x] = True
        
        mask_faces = ~np.any(np.isin(r_faces,np.argwhere(~mask_verts)), axis=1)
        remap_faces = np.arange(r_verts.shape[0])-np.cumsum(~mask_verts)
        
        mesh2 = tvtk.PolyData(points=r_verts[mask_verts,:], polys=remap_faces[r_faces[mask_faces,:]])
        
        dist = tvtk.DistancePolyDataFilter()
        
        configure_port_input_data(dist, 0, mesh1)
        configure_port_input_data(dist, 1, mesh2)
        dist.signed_distance = 0
        dist.compute_second_distance = 1
        dist.update()
        
        ts.append(np.concatenate((dist.output.point_data.scalars.to_array(),dist.get_output(1).point_data.scalars.to_array())).mean())
        ss.append(np.concatenate((dist.output.point_data.scalars.to_array(),dist.get_output(1).point_data.scalars.to_array())).std())


        return sum(ts) / len(ts), np.sqrt(sum([x**2 for x in ss]) / len(ss))
        
    #%%
    mask_verts = np.zeros((r_verts.shape[0],),dtype=np.bool)
    for x in cc[0]:
         mask_verts[x] = True
    
    mask_faces = ~np.any(np.isin(r_faces,np.argwhere(~mask_verts)), axis=1)
    remap_faces = np.arange(r_verts.shape[0])-np.cumsum(~mask_verts)
    
    mesh1 = tvtk.PolyData(points=r_verts[mask_verts,:], polys=remap_faces[r_faces[mask_faces,:]])
    
    mask_verts = np.zeros((r_verts.shape[0],),dtype=np.bool)
    for x in cc[1]:
        mask_verts[x] = True
    
    mask_faces = ~np.any(np.isin(r_faces,np.argwhere(~mask_verts)), axis=1)
    remap_faces = np.arange(r_verts.shape[0])-np.cumsum(~mask_verts)
    
    mesh2 = tvtk.PolyData(points=r_verts[mask_verts,:], polys=remap_faces[r_faces[mask_faces,:]])
    
    dist = tvtk.DistancePolyDataFilter()
    
    configure_port_input_data(dist, 0, mesh1)
    configure_port_input_data(dist, 1, mesh2)
    dist.signed_distance = 0
    dist.compute_second_distance = 1
    dist.update()
    
    
    
    
    # dist2 = tvtk.DistancePolyDataFilter()
    
    # configure_port_input_data(dist2, 0, mesh2)
    # configure_port_input_data(dist2, 1, mesh1)
    # dist2.signed_distance = 0
    # dist2.update()
    
    # from mayavi.sources.api import VTKDataSource
    # from mayavi.api import Engine
    # from mayavi.modules.surface import Surface 
    # src = VTKDataSource(data = dist.output)
    # src2 = VTKDataSource(data = dist2.output)
    
    # e = Engine()
    # e.start()
    # s = e.new_scene()
    # e.add_source(src)
    # e.add_source(src2)
    
    # s = Surface()
    # e.add_filter(s, src)
    # s2 = Surface()
    # e.add_filter(s2, src2)
    
    # s.module_manager.scalar_lut_manager.data_range = [0,5]
    # s2.module_manager.scalar_lut_manager.data_range = [0,5]
    
    
    
    
    return np.concatenate((dist.output.point_data.scalars.to_array(),dist.get_output(1).point_data.scalars.to_array())).mean(), np.concatenate((dist.output.point_data.scalars.to_array(),dist.get_output(1).point_data.scalars.to_array())).std()
