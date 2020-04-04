import cv2
import numpy as np
import math
import torch.utils.data.dataset
import random

class SpiralDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, img_sz=[125, 125]):
        self.cache = dict()
        self.img_sz = img_sz
        self.len = num_samples
        self.dim = [1, img_sz[1], img_sz[0]]
        self.min_pts = 0

    def get_image_from_points(self, pts):
        x = self.pts2image(pts).astype(np.float32)
        return x, x

    def __getitem__(self, index):
        if index in self.cache.keys():
            return self.cache[index]

        num_spirals = random.randint(2, 8)
        min_stops = 1 + ( self.min_pts - 1) // (num_spirals -1)
        min_stops = max(min_stops, 5)
        max_stops = 10 if min_stops < 11 else min_stops + 2
        stops_per_spiral = random.randint(min_stops, max_stops)
        title, pts = self.spiral(num_spirals = num_spirals,
                                 stops_per_spiral = stops_per_spiral,
                                 start_theta_deg = random.randint(0, 180),
                                 radii = [random.random() * 0.4,  random.random() * 0.4 + 0.7])
        x, y = self.get_image_from_points(pts)
        self.cache[index] = (title, x, y)
        return (title, x, y)

    def __len__(self):
        return self.len

    def polar2cart(self, r, theta, i):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, theta, i

    def pts2image(self, pts, connecting_line=True):
        
        def pt2image(pt0):
            # scale and shift: (-1,1) --> (0, im_sz) 
            pt = (pt0[0] * self.img_sz[0]/2 + self.img_sz[0]/2,\
                  pt0[1] * self.img_sz[1]/2 + self.img_sz[1]/2 )
            return (int(pt[0]), int(self.img_sz[1]-pt[1])) # flip the y axis, screen space

        image = np.zeros(self.img_sz, np.uint8)
        pt0 = pt2image(pts[0])
        for i in range(0, len(pts)):
            pt1 = pt2image(pts[i])
            cv2.circle(image, pt0, 1, 255, thickness=2)
            cv2.circle(image, pt1, 1, 255, thickness=2)
            if connecting_line:
                cv2.line(image, pt0, pt1, 255)
            pt0 = pt1

        return np.expand_dims(image, 0).astype(np.float32)

    def spiral(self, num_spirals, stops_per_spiral,
               radii, start_theta_deg):
        theta = start_theta_deg * math.pi / 180
        detla_theta = 2 * math.pi / stops_per_spiral
        radii = np.linspace(radii[0], radii[1], num_spirals + 1)
        pts = [self.polar2cart(radii[0], theta, 0)]
        for sp in range(0, num_spirals - 1):
            for step in range(0, stops_per_spiral):
                theta = theta + detla_theta
                frac = float(step) / stops_per_spiral
                radius = radii[sp] * (1 - frac) + radii[sp+1] * frac
                pt = self.polar2cart(radius, theta, len(pts)) 
                pts.append(pt)

        title = "Sp%d-Pts%d-R[%1.2f-%1.2f]" %(num_spirals, stops_per_spiral, radii[0], radii[-1])
        return title, pts

class SpiralDatasetAsym(SpiralDataset):
    def __init__(self, num_samples, img_sz=[125, 125], split = 2/3):
        super().__init__(num_samples, img_sz)
        self.split  = split

    def get_image_from_points(self, pts):
       split = int(self.split * len(pts))
       x = self.pts2image(pts[:split])
       y = self.pts2image(pts)
       return x, y

class SpiralDatasetTCHW(SpiralDatasetAsym):
    def __init__(self, num_samples, img_sz=[125, 125], split = 2/3):
        super().__init__(num_samples, img_sz, split)
        self.seq_len = 9
        self.t_out = int(self.seq_len * (1 - self.split))
        self.t_in = int(self.seq_len * self.split)
        self.min_pts = self.seq_len
        self.dim = [self.t_in, 1, img_sz[1], img_sz[0]]

    def get_image_from_points(self, pts):
        theta_grp = np.linspace(pts[0][2], pts[-1][2], self.seq_len + 1)
        theta_idx = 1
        pt_list, this_list = [], []

        for pt in pts:
            this_list.append(pt)
            if pt[2] >= theta_grp[theta_idx]:
                theta_idx = theta_idx + 1
                this_list = [this_list[-1]]
                pt_list.append(this_list)

        im_list = [self.pts2image(l) for l in pt_list]

        while len(im_list) < self.seq_len:
            im_list = [np.zeros([1] + self.img_sz, np.float32)] + im_list

        split = int(self.split * len(im_list))
        x = np.stack(tuple(im_list[:split]))
        y = np.stack(tuple(im_list[split:]))
        return x, y