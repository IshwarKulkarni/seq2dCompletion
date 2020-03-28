import cv2
import numpy as np
import math
import torch.utils.data.dataset
import random

class SpiralDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, img_sz=[125, 125], float_type='half'):
        self.cache = dict()
        self.img_sz = img_sz
        self.dim = [1, img_sz[1], img_sz[0]]
        self.len = num_samples
        assert float_type in ['half', 'full']
        self.float_type = np.float16 if float_type == 'half' else np.float32

    def __getitem__(self, index):
        if index not in self.cache.keys():
            img = self.spiral(as_images=True, num_spirals=random.randint(2,8),\
                              stops_per_spiral=random.randint(4,10),\
                              start_theta_deg=0,
                              radii= [random.random() * 0.4,  random.random() * 0.4  + 0.7])
            self.cache[index] = img
            return img
        return self.cache[index]

    def __len__(self):
        return self.len

    def polar2cart(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def pts2image(self, pts, connecting_line=True):
        
        def pt2image(pt0):
            # scale and shift: (-1,1) --> (0, im_sz) 
            pt = (pt0[0] * self.img_sz[0]/2 +  self.img_sz[0]/2,\
                  pt0[1] * self.img_sz[1]/2 +  self.img_sz[1]/2 )
            return (int(pt[0]), int(self.img_sz[1]-pt[1])) # flip the y axis, screen space

        image = np.zeros(self.img_sz, np.uint8)
        for i in range(1, len(pts)):
            pt0 = pt2image(pts[i-1])
            pt1 = pt2image(pts[i])
            cv2.circle(image, pt0, 1, 1, thickness=2)
            cv2.circle(image, pt1, 1, 1, thickness=2)
            if connecting_line:
                cv2.line(image, pt0, pt1, 1)

        return np.expand_dims(image, 0)

    def spiral(self, as_images, num_spirals, stops_per_spiral,\
               radii, start_theta_deg):
        theta = start_theta_deg * math.pi / 180
        detla_theta = 2 * math.pi / stops_per_spiral
        radii = np.linspace(radii[0], radii[1], num_spirals)
        pts = [self.polar2cart(radii[0], theta)]
        for sp in range(0, num_spirals-1):
            for step in range(0, stops_per_spiral):
                theta = theta + detla_theta
                frac = float(step) / stops_per_spiral
                radius = radii[sp] * (1 - frac) + radii[sp+1] * frac
                pt = self.polar2cart(radius, theta) 
                pts.append(pt)
        

        title = "Sp%d-Pts%d-R[%1.2f-%1.2f]" %(num_spirals, stops_per_spiral, radii[0], radii[-1])
        if as_images:
            return title, (self.pts2image(pts)).astype(self.float_type)

        return title, pts
