import random
import numpy as np
from skimage.transform import estimate_transform
from itertools import combinations
from imageio import imread
import pickle
from utils import extract_paths_from_deep_dict, get_from_deep_dict
from collections import namedtuple
import cv2
from augment_color import augment_color
from parameters import params
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

Sample = namedtuple('Sample', 'img pose')


# several functions are adapted from https://github.com/AliaksandrSiarohin/pose-gan/

# UTILS
def give_name_to_keypoints(array, joint_order):
    array = array.T
    res = {}
    for i, name in enumerate(joint_order):
        res[name] = array[i]
    return res


def compute_st_distance(kp):
    st_distance1 = np.sum((kp['rhip'] - kp['rsho']) ** 2)
    st_distance2 = np.sum((kp['lhip'] - kp['lsho']) ** 2)
    return np.sqrt((st_distance1 + st_distance2) / 2.0)


def estimate_polygon(fr, to, st, inc_to, inc_from, p_to, p_from):
    fr = fr + (fr - to) * inc_from
    to = to + (to - fr) * inc_to

    norm_vec = fr - to
    norm_vec = np.array([-norm_vec[1], norm_vec[0]])
    norm = np.linalg.norm(norm_vec)
    if norm == 0:
        return np.array([
            fr + 1,
            fr - 1,
            to - 1,
            to + 1,
        ])
    norm_vec = norm_vec / norm
    return np.array([
        fr + st * p_from * norm_vec,
        fr - st * p_from * norm_vec,
        to - st * p_to * norm_vec,
        to + st * p_to * norm_vec
    ])


def get_array_of_points(kp, names):
    return np.array([kp[name] for name in names])


#
# MASKS
#
def create_part_masks(pose, joint_order):
    pose = pose.copy()
    pose[[0, 1]] = pose[[1, 0]]
    kp_o = give_name_to_keypoints(pose, joint_order)
    st = np.sqrt(max(sum((kp_o['lsho'] - kp_o['rhip']) ** 2), sum((kp_o['rsho'] - kp_o['lhip']) ** 2)))
    image_size = params['image_size']
    size = params['volume_size']
    pose[:2] = pose[:2] / image_size * size
    pose[2] = pose[2] / image_size * params['depth']
    kp = give_name_to_keypoints(pose, joint_order)

    def compute_ellipse(widths):
        ellipse = np.zeros(2 * (widths.astype(np.int16) + 1) + 1)
        for i in range(ellipse.shape[0]):
            for j in range(ellipse.shape[1]):
                for k in range(ellipse.shape[2]):
                    if sum([(x - w) ** 2 / (w + 1) ** 2 for x, w in zip([i, j, k], widths)]) <= 1:
                        ellipse[i, j, k] = 1
        return ellipse

    def draw_thick_line(v, p1, p2, widths):
        widths = np.array(widths)
        ellipse = compute_ellipse(widths)
        widths = widths.astype(np.int16) + 1
        p1 = np.copy(p1)
        d = p2 - p1
        n = max(np.linalg.norm(d), 1)
        dn = d / n

        def fr(i):
            return max(0, p[i] - widths[i])

        def to(i):
            return min(p[i] + widths[i] + 1, v.shape[i])

        def efr(i):
            return max(0, widths[i] - p[i])

        def eto(i):
            return min(ellipse.shape[i] - (p[i] + widths[i] - v.shape[i]) - 1, ellipse.shape[i])

        for i in range(0, int(n)):
            p = np.round(p1 + i * dn).astype(np.int32)
            for j in range(3):
                p[j] = min(v.shape[j] - 1, p[j])
                p[j] = max(0, p[j])

            v[fr(0):to(0), fr(1):to(1), fr(2):to(2)] += ellipse[efr(0):eto(0), efr(1):eto(1), efr(2):eto(2)]
        v[v > 1] = 1

    def draw_mask(vol, points, thickness, end=None):
        for p1, p2 in combinations(points, 2):
            draw_thick_line(vol, p1, p2, [thickness * size / image_size / 2,
                                          thickness * size / image_size / 2,
                                          thickness * params['depth'] / image_size / 2])
        if end is not None:
            draw_thick_line(vol, points[-1], points[-1], [end * size / image_size / 2,
                                                          end * size / image_size / 2,
                                                          end * params['depth'] / image_size / 2])

    masks = np.zeros((size, size, params['depth'], 10), dtype=np.float32)
    draw_mask(masks[..., 0], [kp[joint] for joint in ['rhip', 'lhip', 'lsho', 'rsho']], thickness=0.3 * st)
    if params['dataset'] == 'merged':
        draw_mask(masks[..., 1], [kp[joint] for joint in ['htop', 'head', 'neck']], thickness=0.5 * st)
    elif params['dataset'] in ['iPER', 'fashion3d']:
        center = 0.5 * kp['lear'] + 0.5 * kp['rear']  # head mask is capsule through ear's center
        to_neck = 0.7 * center + 0.3 * kp['neck']  # head mask is stretched in neck-direction
        back_neck = center - (to_neck - center)
        draw_mask(masks[..., 1], [back_neck, to_neck], thickness=0.5 * st)
    else:
        raise ValueError()
    kp['lhip'] = kp['lhip'] + 0.1 * (kp['lhip'] - kp['lsho'])  # move hips down a bit
    kp['rhip'] = kp['rhip'] + 0.1 * (kp['rhip'] - kp['rsho'])
    kp['lwri'] = kp['lwri'] + 0.2 * (kp['lwri'] - kp['lelb'])  # make lower arms contain hands
    kp['rwri'] = kp['rwri'] + 0.2 * (kp['rwri'] - kp['relb'])
    kp['lank'] = kp['lank'] + 0.2 * (kp['lank'] - kp['lkne'])  # make lower legs contain foot
    kp['rank'] = kp['rank'] + 0.2 * (kp['rank'] - kp['rkne'])
    draw_mask(masks[..., 2], [kp[joint] for joint in ['lsho', 'lelb']], thickness=0.2 * st)
    draw_mask(masks[..., 4], [kp[joint] for joint in ['rsho', 'relb']], thickness=0.2 * st)
    draw_mask(masks[..., 3], [kp[joint] for joint in ['lelb', 'lwri']], thickness=0.2 * st, end=0.4 * st)
    draw_mask(masks[..., 5], [kp[joint] for joint in ['relb', 'rwri']], thickness=0.2 * st, end=0.4 * st)
    draw_mask(masks[..., 6], [kp[joint] for joint in ['lhip', 'lkne']], thickness=0.2 * st)
    draw_mask(masks[..., 8], [kp[joint] for joint in ['rhip', 'rkne']], thickness=0.2 * st)
    draw_mask(masks[..., 7], [kp[joint] for joint in ['lkne', 'lank']], thickness=0.2 * st, end=0.4 * st)
    draw_mask(masks[..., 9], [kp[joint] for joint in ['rkne', 'rank']], thickness=0.2 * st, end=0.4 * st)
    if params['2d_3d_warp']:
        masks = np.max(masks, axis=2)
    return masks


#
# TRANSFORM
#
def estimate_transform_params(poses, joint_order):
    if params['2d_3d_warp']:
        return affine_transforms(poses[0], poses[1], joint_order)
    else:
        return helmert_transforms_3d(poses[0], poses[1], joint_order)


# 2D
def affine_transforms(array1, array2, joint_order):
    array1 = array1.copy()[:2]
    array2 = array2.copy()[:2]
    kp1 = give_name_to_keypoints(array1, joint_order)
    kp2 = give_name_to_keypoints(array2, joint_order)

    st1 = compute_st_distance(kp1)
    st2 = compute_st_distance(kp2)

    transforms = []

    body_poly_1 = get_array_of_points(kp1, ['rhip', 'lhip', 'lsho', 'rsho'])
    body_poly_2 = get_array_of_points(kp2, ['rhip', 'lhip', 'lsho', 'rsho'])
    tr = estimate_transform('affine', src=body_poly_2, dst=body_poly_1)

    transforms.append(tr.params)

    head_kp_names = ['neck', 'leye', 'reye', 'nose', 'lear', 'rear', 'lsho', 'rsho']
    head_poly_1 = get_array_of_points(kp1, list(head_kp_names))
    head_poly_2 = get_array_of_points(kp2, list(head_kp_names))
    tr = estimate_transform('affine', src=head_poly_2, dst=head_poly_1)
    transforms.append(tr.params)

    def estimate_join(fr, to, inc_to):
        poly_2 = estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)
        poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
        return estimate_transform('affine', src=poly_2, dst=poly_1).params

    transforms.append(estimate_join('lsho', 'lelb', 0.1))
    transforms.append(estimate_join('lelb', 'lwri', 0.3))

    transforms.append(estimate_join('rsho', 'relb', 0.1))
    transforms.append(estimate_join('relb', 'rwri', 0.3))

    transforms.append(estimate_join('lhip', 'lkne', 0.1))
    transforms.append(estimate_join('lkne', 'lank', 0.3))

    transforms.append(estimate_join('rhip', 'rkne', 0.1))
    transforms.append(estimate_join('rkne', 'rank', 0.3))

    return np.array(transforms).reshape((-1, 9))[..., :-1].astype(np.float32)


# 3D
def helmert_transforms_3d(array1, array2, joint_order):
    array1 = array1.copy()
    array2 = array2.copy()
    kp1 = give_name_to_keypoints(array1, joint_order)
    kp2 = give_name_to_keypoints(array2, joint_order)

    transforms = []

    body_poly_1 = get_array_of_points(kp1, ['rhip', 'lhip', 'lsho', 'rsho'])
    body_poly_2 = get_array_of_points(kp2, ['rhip', 'lhip', 'lsho', 'rsho'])
    transforms.append(estimate_helmert_transform(src=body_poly_2, dst=body_poly_1))

    def estimate_join(fr, to, roll=None):
        if roll is None:
            poly_1 = get_array_of_points(kp1, [fr, to])
            poly_2 = get_array_of_points(kp2, [fr, to])
        else:
            poly_1 = get_array_of_points(kp1, [fr, to, roll])
            poly_2 = get_array_of_points(kp2, [fr, to, roll])
            for poly in [poly_1, poly_2]:
                bone = poly[1] - poly[0]
                roll = poly[2] - poly[1]
                cross = np.cross(bone, roll)
                cross = cross / np.linalg.norm(cross) * np.linalg.norm(roll)
                poly[2] = cross + poly[1]
        return estimate_helmert_transform(src=poly_2, dst=poly_1)

    head_kp_names = ['neck', 'leye', 'reye', 'nose', 'lear', 'rear', 'lsho', 'rsho']
    head_poly_1 = get_array_of_points(kp1, list(head_kp_names))
    head_poly_2 = get_array_of_points(kp2, list(head_kp_names))

    transforms.append(estimate_helmert_transform(src=head_poly_2, dst=head_poly_1))

    transforms.append(estimate_join('lsho', 'lelb', roll='lwri'))
    transforms.append(estimate_join('lwri', 'lelb', roll='lsho'))

    transforms.append(estimate_join('rsho', 'relb', roll='rwri'))
    transforms.append(estimate_join('rwri', 'relb', roll='rsho'))

    transforms.append(estimate_join('lhip', 'lkne', roll='lank'))
    transforms.append(estimate_join('lank', 'lkne', roll='lhip'))

    transforms.append(estimate_join('rhip', 'rkne', roll='rank'))
    transforms.append(estimate_join('rank', 'rkne', roll='rhip'))

    return np.array(transforms)


def estimate_helmert_transform(src, dst):
    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)
    src_c = src - src_center
    dst_c = dst - dst_center
    h = src_c.T @ dst_c
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    d = np.trace(dst_c @ r @ src_c.T) / np.trace(src_c @ src_c.T)
    t = np.expand_dims(dst_center - d * r @ src_center, axis=-1)
    res = np.identity(4, dtype=np.float32)
    res[:3, :3] = d * r
    res[:3, 3:] = t
    return res


def augment_transform_together(imgs, poses, flips):
    size = imgs[0].shape[0]
    alpha = random.uniform(-.1, .1)
    dx = random.uniform(-20, 20)
    dy = random.uniform(-20, 20)
    s = 1 + random.uniform(-0.1, 0.1)
    x = size / 2
    y = size / 2
    z = size / 2
    trans = np.array([[1, 0, 0, x],
                      [0, 1, 0, y],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]])
    trans2 = np.array([[1, 0, 0, -x + dx],
                       [0, 1, 0, -y + dy],
                       [0, 0, 1, -z],
                       [0, 0, 0, 1]])
    rot = np.array([[np.cos(alpha), -np.sin(alpha), 0, 0],
                    [np.sin(alpha), np.cos(alpha), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    scale = np.array([[s, 0, 0, 0],
                      [0, s, 0, 0],
                      [0, 0, s, 0],
                      [0, 0, 0, 1]])
    transform = trans @ rot @ scale @ trans2

    flip = np.random.rand() < 0.5

    for i in range(len(imgs)):
        pose = poses[i]
        img = imgs[i]

        augmented = np.ones((pose.shape[0], pose.shape[1] + 1))
        augmented[:pose.shape[0], :pose.shape[1]] = pose
        augmented = (transform @ augmented.T).T
        new_pose = augmented[:, :-1]

        if params['dataset'] == 'merged':
            new_img = cv2.warpAffine(img, transform[[[0], [1]], [0, 1, 3]], (size, size),
                                     borderMode=cv2.cv2.BORDER_CONSTANT, borderValue=(1, -1, 1))
        elif params['dataset'] in ['iPER', 'fashion3d']:
            new_img = cv2.warpAffine(img, transform[[[0], [1]], [0, 1, 3]], (size, size),
                                     borderMode=cv2.BORDER_REPLICATE)
        else:
            raise ValueError()
        if flip:
            new_img = np.flip(new_img, axis=1)
            new_pose[:, 0] = 256 - new_pose[:, 0]
            new_pose = new_pose[flips]

        poses[i] = new_pose.astype(np.float32)
        imgs[i] = new_img

    return imgs, poses


class Dataset:
    def __init__(self, name, persondepth, joint_order, valid, test, deterministic=False, with_to_masks=False):
        """Dataset class.

        Args:
          name: name of the dataset, used for the dataset path
          persondepth: how many subfolders of the dataset decide the person / clothing layout, if a single file of the
            dataset has for example a path `person/clothing/action/0103.png` the persondepth is 2, since every file
            after the first 2 folders belongs to the same person / clothing layout
          joint_order: a list with the order of joints, these are used for estimating transfomations and creating masks
          valid: a list of all persons of the validation set, given as `person-clothing`
          test: a list of all persons of the test set, given as `person-clothing`
          deterministic: usually the dataset resturns a random sample, this enforces the same order every time
          with_to_masks: whether or not the dataset should also return masks for the target pose
        """
        print('initialize', name, 'dataset')
        self.name = name
        self.poses = self.init_poses()
        self.joint_order = joint_order
        self.with_to_masks = with_to_masks

        self.train, self.valid, self.test = self.init_selectable(persondepth, valid, test)
        if deterministic:
            random.seed(0)
        else:
            random.seed()

        self.flips = []
        for i, joint in enumerate(self.joint_order):
            if joint.startswith('l'):
                other = 'r' + joint[1:]
                if other in self.joint_order:
                    self.flips.append(self.joint_order.index(other))
                else:
                    self.flips.append(i)
            elif joint.startswith('r'):
                other = 'l' + joint[1:]
                if other in self.joint_order:
                    self.flips.append(self.joint_order.index(other))
                else:
                    self.flips.append(i)
            else:
                self.flips.append(i)

    def init_selectable(self, persondepth, valid, test):
        selectable = {}
        keys_list = extract_paths_from_deep_dict(self.poses)
        keys_list = [[str(key) for key in keys] for keys in keys_list]
        for keys in keys_list:
            key = '-'.join(keys[:persondepth])
            if key not in selectable:
                selectable[key] = []
            selectable[key].append(keys)

        singles = []
        for person in selectable:
            if len(selectable[person]) < 2:
                singles.append(person)
        for person in singles:
            selectable.pop(person)

        valid = set(valid)
        test = set(test)
        tr = {}
        va = {}
        te = {}
        for person in selectable:
            if person in valid:
                va[person] = selectable[person]
                if params['with_valid']:
                    tr[person] = selectable[person]
            elif person in test:
                te[person] = selectable[person]
            else:
                tr[person] = selectable[person]
        return tr, va, te

    def init_poses(self):
        pose_file = params['data_dir'] + '/' + self.name + '/poses.pkl'
        with open(pose_file, 'rb') as f:
            poses = pickle.load(f)
        return poses

    def next_train_sample(self):
        while True:
            yield self.uncached_sample(self.train, train=True)

    def next_valid_sample(self):
        while True:
            yield self.uncached_sample(self.valid)

    def next_test_sample(self):
        while True:
            yield self.uncached_sample(self.test)

    def uncached_sample(self, selectable, train=False):
        person = random.choice(list(selectable.keys()))
        if len(selectable[person]) <= 2:
            samples = selectable[person]
        else:
            samples = random.sample(selectable[person], 2)
        if len(samples) == 1:
            samples.append(samples[0])
        fr = self.load(samples[0])
        to = self.load(samples[1])
        return self.get_sample_from_loaded(fr, to, train)

    def get_sample_from_loaded(self, fr, to, train):
        imgs = np.concatenate([fr.img, to.img])
        if params['augment_color'] and train:
            imgs = augment_color(imgs, random)
        splits = np.vsplit(imgs, 2)
        fr_img, to_img = splits[0], splits[1]
        fr_pose, to_pose = fr.pose, to.pose
        if params['augment_transform'] and train:
            fr_pose, to_pose = np.transpose(fr_pose), np.transpose(to_pose)
            (fr_img, to_img), (fr_pose, to_pose) = augment_transform_together([fr_img, to_img], [fr_pose, to_pose],
                                                                              self.flips)
            fr_pose, to_pose = np.transpose(fr_pose), np.transpose(to_pose)
            to_pose[2] += random.uniform(-.5, .5)
        fr_masks = create_part_masks(fr_pose, self.joint_order)

        transform_params = estimate_transform_params([fr_pose, to_pose], self.joint_order)

        if self.with_to_masks:
            to_masks = create_part_masks(to_pose, self.joint_order)
            return fr_img, to_img, fr_masks, to_masks, transform_params, fr_pose, to_pose
        else:
            return fr_img, to_img, fr_masks, transform_params, fr_pose, to_pose

    def load(self, keys):

        img = '/'.join([params['data_dir'], self.name, 'images'] + keys) + '.png'
        # print(img)
        img = np.array(imread(img), dtype=np.float32)
        img = img / 127.5 - 1

        pose = get_from_deep_dict(self.poses, keys).copy().astype(np.float32)

        pose[:, 2] += params['image_size'] / 2  # move z coordinate to range (0, image_size)

        img, pose = img.copy(), pose.copy()

        pose = np.transpose(pose)

        return Sample(img=img, pose=pose)
