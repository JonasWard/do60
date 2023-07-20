from typing import List
from math import sin, cos, pi


ARC_RESOLUTION = 8


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value):
        return Vector(self.x * value, self.y * value, self.z * value)

    def __truediv__(self, value):
        return Vector(self.x / value, self.y / value, self.z / value)

    @staticmethod
    def Origin():
        return Vector(0, 0, 0)

    @staticmethod
    def Z(value=1):
        return Vector(0, 0, value)

    @staticmethod
    def get_center(pts):
        c_p = Vector.Origin()
        for b_p in pts:
            c_p += b_p

        c_p /= len(pts)

        return c_p


class Mesh:
    def __init__(self, name: str, vertices: List[Vector], faces: List[tuple], normals=None, uvs=None, texture=None):
        self.name = name
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.uvs = uvs
        self.texture = texture

    @staticmethod
    def join_2_meshes(mesh_a, mesh_b, new_name=None):
        # create a new mesh
        new_name = new_name if new_name != None else mesh_a.name + '_' + mesh_b.name
        new_vertices = mesh_a.vertices + mesh_b.vertices
        new_faces = mesh_a.faces + \
            [tuple(f_i + len(mesh_a.vertices) for f_i in face)
             for face in mesh_b.faces]
        new_normals = mesh_a.normals + \
            mesh_b.normals if mesh_a.normals != None and mesh_b.normals != None else None
        new_uvs = mesh_a.uvs + mesh_b.uvs if mesh_a.uvs != None and mesh_b.uvs != None else None
        new_texture = mesh_a.texture + \
            mesh_b.texture if mesh_a.texture != None and mesh_b.texture != None else None

        mesh = Mesh(new_name,
                    new_vertices,
                    new_faces,
                    new_normals,
                    new_uvs,
                    new_texture)

        return mesh

    @staticmethod
    def join_meshes(meshes, new_name=None):
        if len(meshes) == 0:
            return None
        if len(meshes) == 1:
            return meshes[0]

        mesh = Mesh.Empty()
        for i in range(0, len(meshes)):
            mesh = Mesh.join_2_meshes(mesh, meshes[i])

        mesh.name = new_name

        return mesh

    def as_obj(self):
        create_obj(self)

    @staticmethod
    def Empty():
        return Mesh('', [], [])


def create_obj(mesh: Mesh):
    # create a file
    with open('/Users/jonasvandenbulcke/Documents/reps/do60/' + mesh.name + '.obj', 'w') as f:
        # write the vertices
        for v in mesh.vertices:
            f.write('v {} {} {}\n'.format(v.x, v.z, v.y))
        if mesh.uvs != None:
            for uv in mesh.uvs:
                f.write('vt {} {}\n'.format(uv[0], uv[1]))
        if mesh.normals != None:
            for n in mesh.normals:
                f.write('vn {} {} {}\n'.format(n.x, n.y, n.z))

        # write the faces
        for face in mesh.faces:
            f.write('f')
            for f_i in face:
                f.write(' {}/{}/{}'.format(1 + f_i, 1 + f_i, 1 + f_i))
            f.write('\n')

    return


def create_cube():
    vertices = [
        Vector(0, 0, 0),
        Vector(0, 0, 1),
        Vector(0, 1, 0),
        Vector(0, 1, 1),
        Vector(1, 0, 0),
        Vector(1, 0, 1),
        Vector(1, 1, 0),
        Vector(1, 1, 1),
    ]

    faces = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 3, 7, 5]
    ]

    normals = [
        Vector(0, 0, -1),
        Vector(0, 0, 1),
        Vector(0, -1, 0),
        Vector(0, 1, 0),
        Vector(-1, 0, 0),
        Vector(1, 0, 0)
    ]

    uvs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    textures = [
        'texture.png'
    ]

    name = 'cube'

    mesh = Mesh(name, vertices, faces, normals, uvs, textures)

    mesh.as_obj()

    return


def create_arc(p_b, d_a, d_b):
    angle_step = pi * .5 / ARC_RESOLUTION
    pts = []
    for i in range(ARC_RESOLUTION + 1):
        a = d_a * cos(angle_step * i)
        b = d_b * sin(angle_step * i)
        pts.append(p_b + a + b)

    return pts


def loft_pts(pts_a, pts_b, name='loft_mesh'):
    faces = [(i, i + 1, len(pts_a) + 1 + i, len(pts_a) + i)
             for i in range(len(pts_a) - 1)]

    return Mesh(name, pts_a + pts_b, faces)


def fit_arc_pts(p_a, p_b, d_a_a, d_b_a, d_a_b, d_b_b):
    pts_a = create_arc(p_a, d_a_a, d_b_a)
    pts_b = create_arc(p_b, d_a_b, d_b_b)

    return pts_a, pts_b


def fit_arc(p_a, p_b, d_a_a, d_b_a, d_a_b, d_b_b):
    return loft_pts(*fit_arc_pts(p_a, p_b, d_a_a, d_b_a, d_a_b, d_b_b))


def create_voxel(base_pts, h0, h1):
    c_p = Vector.get_center(base_pts)

    vertices = [Vector(v.x, v.y, h0) for v in base_pts] + [Vector(c_p.x, c_p.y, h0)] + \
        [Vector(v.x, v.y, h1) for v in base_pts] + [Vector(c_p.x, c_p.y, h1)]
    faces = []

    c_p_0 = len(base_pts)
    c_p_1 = len(base_pts) * 2 + 1

    for i in range(len(base_pts)):
        p_a_0_i = i
        p_b_0_i = (i + 1) % len(base_pts)
        p_a_1_i = i + len(base_pts) + 1
        p_b_1_i = (i + 1) % len(base_pts) + len(base_pts) + 1

        faces.append([p_a_0_i, p_b_0_i, c_p_0])
        faces.append([p_b_1_i, p_a_1_i, c_p_1])
        faces.append([p_a_0_i, p_a_1_i, p_b_1_i, p_b_0_i])

    return Mesh('voxel', vertices, faces)


def create_polygon(radius, n_sides):
    angle_step = 2 * pi / n_sides
    pts = []
    for i in range(n_sides):
        a = Vector(cos(angle_step * i), sin(angle_step * i), 0) * radius
        pts.append(a)

    return pts


def subdivide_grids_n_times(grid, n):
    for i in range(n):
        n_grid = []
        for p in grid:
            n_grid.extend(grid_sub_d(p))
        grid = n_grid

    return grid


def grid_sub_d(pts):
    c_p = Vector.get_center(pts)

    m_pts = []

    for i in range(len(pts)):
        p_a = pts[i]
        p_b = pts[(i + 1) % len(pts)]
        b_p_a = (p_a + p_b) * .5

        m_pts.append(b_p_a)

    grid = []
    for i in range(len(pts)):
        p_00 = pts[i]
        p_01 = m_pts[i]
        p_10 = m_pts[(i - 1) % len(pts)]
        p_11 = c_p

        grid.append([p_00, p_01, p_11, p_10])

    return grid


def polygon_mesh(pts, flip=False):
    vertices = pts + [Vector.get_center(pts)]
    faces = []
    for i in range(len(pts)):
        faces.append([i, (i - 1) % len(pts), len(pts)])

    if flip:
        faces = [face[::-1] for face in faces]

    return Mesh('polygon_mesh', vertices, faces)


def arc_side_profile(p00, p01, p10, p11, arc_pts, flip=False):
    vertices = [p00, p01, p10, p11] + arc_pts
    faces = [
        [0, 1, len(vertices) - 1],  # top triangle
        [2, 0, 6, 5],
        [2, 5, 4, 3]
    ]

    for i in range(6, len(vertices) - 1):
        faces.append([i, 0, i + 1])

    if flip:
        faces = [face[::-1] for face in faces]

    return Mesh('arc_side_profile', vertices, faces)


def arc_voxel(base_pts, h0, h1, b0, b1, s0, s1):
    c_p = Vector.get_center(base_pts)
    r_top_arc = (h1 - h0 - b1 - b0) * s1
    d_b_b = Vector.Z(r_top_arc)

    mshs = []

    z_a = h1 - b1 - r_top_arc

    # create the fit arcs
    for i in range(len(base_pts)):
        p_a = base_pts[i]
        p_b = base_pts[(i + 1) % len(base_pts)]
        b_p_a = (p_a + p_b) * .5

        d_a_a_1 = (p_a - b_p_a) * s0
        d_a_b_2 = (p_b - b_p_a) * s0
        d_a_a_2 = (p_b - c_p) * s0
        d_a_b_1 = (p_a - c_p) * s0

        v0 = Vector(b_p_a.x, b_p_a.y, z_a)
        v1 = Vector(c_p.x, c_p.y, z_a)

        pts_a_a, pts_b_a = fit_arc_pts(
            v0, v1, d_a_a_1, d_b_b, d_a_b_1, d_b_b)
        pts_a_b, pts_b_b = fit_arc_pts(
            v1, v0, d_a_a_2, d_b_b, d_a_b_2, d_b_b)

        # bottom lofts

        pts_a_a = [Vector(pts_a_a[-1].x, pts_a_a[-1].y, h0 + b0),
                   Vector(pts_a_a[0].x, pts_a_a[0].y, h0 + b0)] + pts_a_a
        pts_b_a = [Vector(pts_b_a[-1].x, pts_b_a[-1].y, h0 + b0),
                   Vector(pts_b_a[0].x, pts_b_a[0].y, h0 + b0)] + pts_b_a
        pts_a_b = [Vector(pts_a_b[-1].x, pts_a_b[-1].y, h0 + b0),
                   Vector(pts_a_b[0].x, pts_a_b[0].y, h0 + b0)] + pts_a_b
        pts_b_b = [Vector(pts_b_b[-1].x, pts_b_b[-1].y, h0 + b0),
                   Vector(pts_b_b[0].x, pts_b_b[0].y, h0 + b0)] + pts_b_b

        mshs.append(loft_pts(pts_a_a, pts_b_a, 'bottom_loft_a'))
        mshs.append(loft_pts(pts_a_b, pts_b_b, 'bottom_loft_b'))

        # side profiles
        mshs.append(arc_side_profile(Vector(p_a.x, p_a.y, h1), Vector(
            b_p_a.x, b_p_a.y, h1), Vector(p_a.x, p_a.y, h0), Vector(b_p_a.x, b_p_a.y, h0), pts_a_a))
        mshs.append(arc_side_profile(Vector(p_b.x, p_b.y, h1), Vector(b_p_a.x, b_p_a.y, h1), Vector(
            p_b.x, p_b.y, h0), Vector(b_p_a.x, b_p_a.y, h0), pts_b_b, True))

    # close top
    mshs.append(polygon_mesh([Vector(p.x, p.y, h1) for p in base_pts], False))
    mshs.append(polygon_mesh([Vector(p.x, p.y, h0) for p in base_pts], True))

    msh = Mesh.join_meshes(mshs, 'arc_voxel')

    return msh


def shape_generator():

    sides = 7
    mult = .8
    inset = .05
    side_mult = .2
    # create polygon
    plg = create_polygon(200, sides)

    column_cells = []
    center_cell = []
    arc_cells = []

    for i in range(sides):
        p_0 = plg[(i - 1) % sides]
        p_1 = plg[i]
        p_2 = plg[(i + 1) % sides]
        p_1_bis = p_1 * mult
        p_2_bis = p_2 * mult

        center_cell.append(p_1_bis)

        d_0 = (p_1 - p_0)
        d_1 = (p_2 - p_1)

        p_int_0 = Vector(-d_0.y, d_0.x, 0) * inset + \
            p_0 + d_0 * (1 - side_mult)
        p_int_1 = Vector(-d_1.y, d_1.x, 0) * inset + p_1 + d_1 * side_mult
        p_int_2 = Vector(-d_1.y, d_1.x, 0) * inset + \
            p_1 + d_1 * (1 - side_mult)

        column_cells.append([p_int_0, p_1, p_int_1, p_1_bis])
        arc_cells.append([p_int_1, p_int_2, p_2_bis, p_1_bis])

    single_arc_grid = arc_cells + [center_cell]

    # subdivide a couple of times
    column_cells = subdivide_grids_n_times(column_cells, 1)
    arc_cells = subdivide_grids_n_times(arc_cells, 2)
    center_cell = subdivide_grids_n_times([center_cell], 2)

    hs = [0, 80, 60, 40, 30, 20, 30, 40, 60, 20, 10, 15, 10]
    # partial sums of hs
    height_map = [[sum(hs[:i]), sum(hs[:i + 1])] for i in range(1, len(hs))]

    center_height = height_map[len(hs) - 5][0]

    mshes = []
    for [h0, h1] in height_map:
        mshes.extend([arc_voxel(vs, h0, h1, 2.5, 4, .6, .8)
                     for vs in column_cells])

        if (h0 >= center_height):
            mshes.extend([arc_voxel(vs, h0, h1, 2.5, 4, .6, .8)
                         for vs in center_cell])
            mshes.extend([arc_voxel(vs, h0, h1, 2.5, 4, .6, .8)
                         for vs in arc_cells])

    mshes.extend([arc_voxel(vs, 0, center_height, 2.5, 4, .6, .2)
                 for vs in single_arc_grid])

    # create the voxel
    shape_mesh = Mesh.join_meshes(mshes)
    shape_mesh.name = 'them_shape'
    shape_mesh.as_obj()


if __name__ == '__main__':

    shape_generator()
