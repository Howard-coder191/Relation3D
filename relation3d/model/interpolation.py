def trilinear_interpolation(self, object_agents, points, grid_size=(16, 16, 4)):
    N, C = points.shape[0], object_agents.shape[1]
    gx, gy, gz = grid_size

    points_scaled = points * torch.tensor([gx-1, gy-1, gz-1], dtype=torch.float32)

    x0 = points_scaled[:, 0].floor().long()
    y0 = points_scaled[:, 1].floor().long()
    z0 = points_scaled[:, 2].floor().long()
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    x0 = torch.clamp(x0, 0, gx-2)
    y0 = torch.clamp(y0, 0, gy-2)
    z0 = torch.clamp(z0, 0, gz-2)
    x1 = torch.clamp(x1, 1, gx-1)
    y1 = torch.clamp(y1, 1, gy-1)
    z1 = torch.clamp(z1, 1, gz-1)

    xd = (points_scaled[:, 0] - x0.float()).unsqueeze(1)
    yd = (points_scaled[:, 1] - y0.float()).unsqueeze(1)
    zd = (points_scaled[:, 2] - z0.float()).unsqueeze(1)

    object_agents_grid = object_agents.view(gx, gy, gz, C)
    
    v000 = object_agents_grid[x0, y0, z0]
    v001 = object_agents_grid[x0, y0, z1]
    v010 = object_agents_grid[x0, y1, z0]
    v011 = object_agents_grid[x0, y1, z1]
    v100 = object_agents_grid[x1, y0, z0]
    v101 = object_agents_grid[x1, y0, z1]
    v110 = object_agents_grid[x1, y1, z0]
    v111 = object_agents_grid[x1, y1, z1]

    c00 = v000 * (1 - xd) + v100 * xd
    c01 = v001 * (1 - xd) + v101 * xd
    c10 = v010 * (1 - xd) + v110 * xd
    c11 = v011 * (1 - xd) + v111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    object_queries = c0 * (1 - zd) + c1 * zd

    return object_queries