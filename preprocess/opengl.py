import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image, ImageOps
import math
import lmdb
import argparse
import os

vertices = np.array([
    [-0.25, 0.0, 2],
    [0.25, 0.0, 2],
    [0.0, 0.0, 4],
    [-0.25, 0.125, 2],
    [0.25, 0.125, 2],
    [0, 0.125, 4],
])

colors = np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0]
])

indices_tri = np.array([
    [2, 0, 1],
    [5, 3, 4],
])

indices_quad = np.array([
    [3, 0, 1, 4],
    [5, 2, 0, 3],
    [4, 1, 2, 5]
])


def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, 1.0, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


# 渲染箭头
def render_arrow(directions):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    radius = 10
    alpha = 0

    gluLookAt(-math.sin(alpha * (math.pi / 180)) * radius, -4.0, -math.cos(alpha * (math.pi / 180)) * radius, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)

    for direction in directions:
        glRotate(-direction['angle'], 0.0, 1.0, 0.0)
        if direction['color'] == 'blue':
            glColor3f(0.0, 0.0, 1.0)
        elif diretion['color'] == 'red':
            glColor3f(1.0, 0.0, 0.0)
        glDrawElements(GL_TRIANGLES, len(indices_tri.flatten()), GL_UNSIGNED_INT, indices_tri)
        glDrawElements(GL_QUADS, len(indices_quad.flatten()), GL_UNSIGNED_INT, indices_quad)
        glRotate(direction['angle'], 0.0, 1.0, 0.0)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glFlush()

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (640, 480), data)
    image = ImageOps.flip(image)  # in my case image is flipped top-bottom for some reason
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def draw_opengl_to_image(directions, image_width, image_height):
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(image_width, image_height)
    window_id = glutCreateWindow("3D Arrow")

    glViewport(0, 0, image_width, image_height)
    init()

    image = render_arrow(directions)

    # glutDisplayFunc(partial(render_arrow, directions))
    # glutMainLoop()

    glutDestroyWindow(window_id)

    return image


def get_nearest_heading(curr_heading, neighbour_angles, go_towards):
    diff = float('inf')
    if go_towards == 'left':
        diff_func = lambda next_heading, curr_heading: (curr_heading - next_heading) % 360
    elif go_towards == 'right':
        diff_func = lambda next_heading, curr_heading: (next_heading - curr_heading) % 360

    for heading in neighbour_angles:
        if heading == curr_heading:
            # don't match to the current heading when turning
            continue
        diff_ = diff_func(int(heading), int(curr_heading))
        if diff_ < diff:
            diff = diff_
            next_heading = heading

    if next_heading is None:
        next_heading = curr_heading

    return next_heading, diff


def ps_images(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out_env = lmdb.open(args.output_dir, map_size=int(1e12))

    out_txn = out_env.begin(write=True)
    for angle in range(360):
        if angle == 0:
            color = 'blue'
        else:
            color = 'red'
        directions = [{'angle': angle, 'color': color}]

        direction_image = draw_opengl_to_image(directions, 640, 480)
        direction_image = cv2.resize(direction_image, (224, 224))
        dx, dy = 0, -65
        MAT = np.float32([[1, 0, dx], [0, 1, dy]])
        direction_image = cv2.warpAffine(direction_image, MAT, (224, 224))

        _, image_bytes = cv2.imencode('.png', direction_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        angle_key = str(angle).encode('ascii')
        out_txn.put(angle_key, image_bytes)

    out_txn.commit()
    out_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and compress pano images.')
    parser.add_argument('--output_dir', default='/home/xyz9911/Datasets/Streetlearn/directions_images', type=str)
    args = parser.parse_args()

    ps_images(args)
