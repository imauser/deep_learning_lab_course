import numpy as np
import scipy.ndimage as nd
import random

from scipy.ndimage.filters import gaussian_filter

class Augmentator():

    @staticmethod
    def augment_images(images, augmentation_factor):
        out = []

        zooms = int(augmentation_factor * random.random() * 1./3.)
        eds = int((augmentation_factor - zooms ) *random.random() * 1./2.)
        rotations = int(augmentation_factor - zooms - eds)
        for i in images:
            for j in range(zooms):
                out.append(Augmentator.augment_keeping_image_size(Augmentator.zoom, i,\
                        zoom=[1, 1+random.random(), 1+random.random()]))
            for j in range(eds):
                out.append(Augmentator.augment_keeping_image_size(\
                        Augmentator.elastic_deformation, i, sigma=2*random.random(),\
                        alpha=4*random.random()));
            for j in range(rotations):
                out.append(Augmentator.augment_keeping_image_size(\
                        Augmentator.rotate, i, angle=random.random()* 360.));

        return out

    @staticmethod
    def rotate(image, angle, axes=(1,2),*args, **kwargs):
        return nd.interpolation.rotate(image, angle, axes, *args, **kwargs)


    @staticmethod
    def zoom(image, zoom=[1,1.5,1.5], *args, **kwargs):
        return nd.interpolation.zoom(image, zoom, *args, **kwargs)

    @staticmethod
    def augment_keeping_image_size(func, image, *args, **kwargs):
        shape = (image.shape[1], image.shape[2])
        new_image = func(image, *args, **kwargs)
        offset = (new_image.shape[1] - shape[0])/2, (new_image.shape[2] - shape[1])/2
        p = new_image[:, offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]]
        return p

    @staticmethod
    def zoom_keeping_image_size(image,*args,**kwargs):
        return Augmentator.augment_keeping_image_size(Augmentator.zoom, image,\
                *args, **kwargs)

    @staticmethod
    def rotate_keeping_image_size(image,*args,**kwargs):
        return Augmentator.augment_keeping_image_size(Augmentator.rotate, image,\
                *args, **kwargs)



    @staticmethod
    def elastic_deformation(image, sigma=1.,alpha=100.):
        """
        Caution only works on 3 layer images
        """

        random_state = np.random.RandomState(None)

        # create a field with new coordinate positions
        shape = (image.shape[1], image.shape[2])
        dx = alpha * nd.filters.gaussian_filter((random_state.rand(*shape)\
                * 2 - 1), sigma, mode="constant")
        dy = alpha * nd.filters.gaussian_filter((random_state.rand(*shape)\
                * 2 - 1), sigma, mode="constant")

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices =  np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # map coordinates does not want multi channel images
        if len(image.shape) == 2:
            return nd.interpolation.map_coordinates(image, indices, order=1).reshape(shape)

        im = []
        for c in range(image.shape[0]):
            im.append(nd.interpolation.map_coordinates(image[c,:,:], indices,\
                    order=1).reshape(shape))

        im = np.array(im)
        return im
