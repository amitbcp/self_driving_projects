

# Imports
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Cluster:
  def __init__(self, dir_path, tag='CG_ClusterExp', flatten=True):
    """
        Constructor for Cluster class
        :param dir_path: Path of directory holding chart images
        :param tag: Tag associated with experiment run (for segregation)
        :return:
        """
    self.exp_tag = tag
    self.allowed_formats = ['jpg', 'jpeg', 'png', 'tiff']
    self.dir_path = os.path.abspath(dir_path)
    self.imgs = self.load_img_data(flatten=flatten)
    self.img_feats = self.analyse_imgs()

  def load_img_data(self, flatten=True):
    """
        Loads all the images (allowed formats) from self.dir_path
        :param flatten: If True, removes alpha channels from image
        :return imgs: List of numpy arrays
        """
    print('[INFO] Loading Image Data')
    img_paths = [
      os.path.join(self.dir_path, path) for path in os.listdir(self.dir_path)
      if path.rsplit('.', 1)[-1] in self.allowed_formats
    ]

    if flatten:
      flat_dir = os.path.join(self.dir_path, 'flattened')
      if not os.path.isdir(flat_dir):
        os.mkdir(flat_dir)

      for path_id in tqdm(range(len(img_paths)), desc='Flattening Images'):
        img_path = img_paths[path_id]
        out_img_path = os.path.join(
          flat_dir,
          img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0] + '.png')

        conv_cmd = ' '.join([
          'convert', img_path, '-alpha remove', '-background white',
          out_img_path
        ])
        os.system(conv_cmd)

        img_paths[path_id] = out_img_path

    imgs = [
      img_np for img_np in [
        cv2.imread(img_path)
        for img_path in tqdm(img_paths, desc='Loading Images as Numpy')
      ] if img_np is not None
    ]

    print('Done.')
    return imgs

  def analyse_imgs(self):
    """
        Compute analytics/features about the loaded data
        :return img_feats: dict of image-data features
        """
    img_feats = {'avg_h': 0, 'avg_w': 0, 'num_imgs': 0}

    img_feats['num_imgs'] = len(self.imgs)
    for img in tqdm(self.imgs, desc='Computing image analytics'):
      h, w, _ = img.shape
      img_feats['avg_h'] += h
      img_feats['avg_w'] += w

    img_feats['avg_h'] = int(img_feats['avg_h'] / img_feats['num_imgs'])
    img_feats['avg_w'] = int(img_feats['avg_w'] / img_feats['num_imgs'])

    print('Done.')
    print(img_feats)
    return img_feats

  def pre_process(self, gray=True, resize=True):
    """
        Pre-processing steps to be applied on raw chart images
        :param gray: If True, converts image to grayscale
        :param resize: If True, resizes the image to data-average
        :return
        """
    print('[INFO] Performing Pre-Processing Steps on Images')
    for idx in tqdm(range(len(self.imgs)), desc='Grayscale and Resize'):
      img = self.imgs[idx]

      if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      if resize:
        img = cv2.resize(img,
                         (self.img_feats['avg_w'], self.img_feats['avg_h']))

      self.imgs[idx] = img

    print('Done.')
    return

  def cluster(self, n_cluster=20, do_tsne=False, vis=False):
    """
        Perform K-means clustering on images
        :param n_cluster: Number of clusters to create
        :param do_tsne: Create TSNE embeddings before clustering
        :param vis: Create visualization of the cluster-center images created (not to be used with TSNE)
        :return cluster_map: dictionary of key-cluster_id and values-list_of_img_objects
        """
    print('[INFO] Performing k-means clustering')
    imgs_np = np.array(self.imgs)
    print('[INFO] Input Data  : {}'.format(imgs_np.shape))
    res_imgs_np = imgs_np.reshape(
      len(self.imgs), self.img_feats['avg_h'] * self.img_feats['avg_w'])

    if do_tsne:
      print('\t>Creating TSNE embeddings')
      tsne = TSNE(n_components=2, init='random', random_state=0)
      img_feats = tsne.fit_transform(res_imgs_np)
    else:
      img_feats = res_imgs_np

    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_jobs=-1)
    clusters = kmeans.fit_predict(img_feats)

    cluster_map = {}
    for im_idx in tqdm(
        range(len(clusters)), desc='Binning images to clusters'):
      pred, img = str(clusters[im_idx]), self.imgs[im_idx]

      if pred not in cluster_map:
        cluster_map[pred] = [img]
      else:
        cluster_map[pred].append(img)

    if vis:
      print('\t>Saving cluster-center visualization')
      n_clusts = len(kmeans.cluster_centers_)
      fig, ax = plt.subplots(1 + (n_clusts / 5), 5)
      centers = kmeans.cluster_centers_.reshape(
        n_clusts, self.img_feats['avg_h'], self.img_feats['avg_w'])
      for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.gray)
      plt.savefig(
        os.path.join(self.dir_path, self.exp_tag + '_' + 'cluster_vis.png'))

    print('Done.')
    return cluster_map

  def save_images(self, images=None, out_path=None):
    """
        Save images (if not provided, self.imgs) to out_path (if provided), else to dir_path/output_<exp_tag>
        :param images: list of image objects
        :param out_path: Path to output directory
        :return:
        """
    if not out_path:
      out_path = os.path.join(self.dir_path, 'output_' + self.exp_tag)

    if not os.path.isdir(out_path):
      os.mkdir(out_path)

    if not images:
      images = self.imgs

    for i, img in tqdm(enumerate(images), desc='Saving images to ' + out_path):
      cv2.imwrite(os.path.join(out_path, str(i) + '.png'), img)

    print('Done.')
    return

  def save_cluster(self, cluster_map, was_tsne=False):
    """
        Saves a cluster_map object as a one-dir-per-cluster structure
        :param cluster_map: dictionary of key-cluster_id and values-list_of_img_objects
        :param was_tsne: Whether or not TSNE was used.
        :return:
        """
    if was_tsne:
      clust_tag = '_'.join(
        ['clusters', str(len(cluster_map)), 'tsne', self.exp_tag])
    else:
      clust_tag = '_'.join(['clusters', str(len(cluster_map)), self.exp_tag])

    clusts_out_dir = os.path.join(self.dir_path, clust_tag)
    if not os.path.isdir(clusts_out_dir):
      os.mkdir(clusts_out_dir)
    else:
      print('Removing previous cluster outputs')
      os.system('rm -rf ' + clusts_out_dir + '/*')

    print('[INFO] Saving clusters to ' + clusts_out_dir)
    for cluster_id in cluster_map:
      clust_path = os.path.join(clusts_out_dir, cluster_id)
      self.save_images(images=cluster_map[cluster_id], out_path=clust_path)

    print('Done.')
    return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='CLI for Image-Clustering Routines')
  parser.add_argument(
    '-dir',
    '--dir',
    required=True,
    help='Path to directory containing Chart Images')
  parser.add_argument(
    '-tag',
    '--tag',
    default='CG_ClusterExp',
    help='Tag to associate with this experiment')
  parser.add_argument(
    '-n_cluster',
    '--n_cluster',
    type=int,
    required=True,
    help='Number of clusters to create')
  parser.add_argument(
    '--flatten',
    action='store_true',
    help='Flatten images to remove alpha channel')
  parser.add_argument(
    '--gray', action='store_true', help='Convert images to grayscale')
  parser.add_argument(
    '--resize', action='store_true', help='Resize images to data-average size')
  parser.add_argument(
    '--tsne',
    action='store_true',
    help='Create TSNE embeddings before clustering')

  args = parser.parse_args()

  operator = Cluster(dir_path=args.dir, tag=args.tag, flatten=args.flatten)
  operator.pre_process(gray=args.gray, resize=args.resize)
  cluster_map = operator.cluster(n_cluster=args.n_cluster, do_tsne=args.tsne)
  operator.save_cluster(cluster_map, was_tsne=args.tsne)
