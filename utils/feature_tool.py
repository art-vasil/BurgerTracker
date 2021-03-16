import cv2
import numpy as np

from settings import GOOD_MATCH_NUMBER

FLANN_INDEX_KDTREE = 0


class KeyPointOffset:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
        search_params = dict(checks=5)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def calc_offset(self, l_img, r_img):

        # extract the keypoints and descriptions
        feature1 = self.get_surf_features(l_img)
        feature2 = self.get_surf_features(r_img)

        # matching the keypoints
        matches = self.flann.knnMatch(feature1['des'], feature2['des'], k=2)

        kps1 = feature1['kp']
        kps2 = feature2['kp']
        # filter the pointes
        matches_mask = [[0, 0] for i in range(len(matches))]
        goods = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.5 * n.distance:
                goods.append((m.trainIdx, m.queryIdx))
                matches_mask[i] = [1, 0]
        # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matches_mask, flags=0)
        # img3 = cv2.drawMatchesKnn(l_img, kps1, r_img, kps2, matches, None, **draw_params)
        # cv2.imshow("matches image", img3)
        # cv2.waitKey()

        if len(goods) >= GOOD_MATCH_NUMBER:

            matched_kps1 = np.float32([kps1[i].pt for (_, i) in goods])
            matched_kps2 = np.float32([kps2[i].pt for (i, _) in goods])

            return matched_kps1, matched_kps2
        else:
            return False, [None, None, None]

        # return len(goods)

    def get_surf_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}


if __name__ == '__main__':
