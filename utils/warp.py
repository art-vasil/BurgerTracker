import numpy as np
import cv2
from scipy import signal


def show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].
    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    #     image_out = np.zeros(image_in.shape)
    #     image_out = cv2.normalize(image_in, image_out, alpha=scale_range[0],
    #                   beta=scale_range[1], norm_type=cv2.NORM_MINMAX)
    image_out = cv2.normalize(image_in, None, alpha=scale_range[0],
                              beta=scale_range[1], norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return image_out


def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel. set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    # assert if image is not in gray scale
    assert len(image.shape) < 3, "input image is not in gray scale"
    # assert if image values are not within range [0.0, 1.0]
    assert (image >= 0).all() and (image <= 1).all(), "input image values are not within range of 0 to 1"

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) / 255

    # check the value range of the output (sobelx) is [0.0, 1.0]
    if not ((sobelx >= 0).all() and (sobelx <= 1).all()):
        sobelx = normalize_and_scale(sobelx, (0, 1))

    return sobelx


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel. set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    # assert if image is not in gray scale
    assert len(image.shape) < 3, "input image is not in gray scale"
    # assert if image values are not within range [0.0, 1.0]
    assert (image >= 0).all() and (image <= 1).all(), "input image values are not within range of 0 to 1"

    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # check the value range of the output (sobely) is [0.0, 1.0]
    if not ((sobely >= 0).all() and (sobely <= 1).all()):
        sobely = normalize_and_scale(sobely, (0, 1))

    return sobely


def reduce_image(image):
    """Reduces an image to half its shape. rounding up:
    example: input image (13, 19) -> output image (7, 10)

    implement it using a convolution-based method using the 5-tap separable filter.

    refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    # assert if image is not in gray scale
    assert len(image.shape) < 3, "input image is not in gray scale"
    # assert if image values are not within range [0.0, 1.0]
    assert (image >= 0).all() and (image <= 1).all(), "input image values are not within range of 0 to 1"

    # blur by 5x5 gaussian, then downsample by half
    img_bd = cv2.GaussianBlur(image, (5, 5), 0, 0)[::2, ::2]

    # check the value range of the output (sobely) is [0.0, 1.0]
    if not ((img_bd >= 0).all() and (img_bd <= 1).all()):
        img_bd = normalize_and_scale(img_bd, (0, 1))

    return img_bd


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image. The pyramid levels (number of images)
    is determined in arg levels.

    This method uses reduce_image() at each level i.e. each level is a reduced image of the lower level and so on

    Images are stored in a list of length equal the number of levels. The first element in the list ([0]) should
    contain the input image. All other levels contain a reduced version of the previous level.

    All images in the pyramid should be floating-point with values in range [0.0, 1.0]

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pyramids = []
    r_image = image.copy()
    for i in range(levels):
        pyramids.append(r_image)
        r_image = reduce_image(r_image)

    return pyramids


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    images = []
    padding = 2
    max_height = 0  # find the max height of all the images
    total_width = 0  # the total width of the images (horizontal stacking)
    for img in img_list:

        images.append(img)
        image_height = img.shape[0]
        image_width = img.shape[1]
        if image_height > max_height:
            max_height = image_height
        # add all the images widths
        total_width += image_width

    final_image = np.ones((max_height, (len(img_list) - 1) * padding + total_width))
    current_x = 0  # keep track of where your current image was last placed in the x coordinate
    for image in images:
        # add an image to the final array and increment the x coordinate
        height = image.shape[0]
        width = image.shape[1]
        final_image[:height, current_x:width + current_x] = image
        # add the padding between the images
        current_x += width + padding

    if not ((final_image >= 0).all() and (final_image <= 1).all()):
        final_image = normalize_and_scale(final_image, (0, 1))

    return final_image


def expand_image(image):
    """Expands (upsamples) an image doubling its width and height.

    Upsample the image by 2 (doubling the size), then blur (filter).
    The filter can be done by implementing a convolution-based
    method using the 5-tap separable filter.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    # assert if image is not in gray scale
    assert len(image.shape) < 3, "input image is not in gray scale"
    # assert if image values are not within range [0.0, 1.0]
    assert (image >= 0).all() and (image <= 1).all(), "input image values are not within range of 0 to 1"

    (width, height) = image.shape
    newimage = np.zeros([width * 2, height * 2])
    newimage[::2, ::2] = image

    param = 0.4
    kernel = np.array([0.25 - param / 2.0, 0.25, param,
                       0.25, 0.25 - param / 2.0])
    kernel = np.outer(kernel, kernel)
    img_out = signal.convolve2d(newimage, kernel, 'same') * 4

    if not ((img_out >= 0).all() and (img_out <= 1).all()):
        img_out = normalize_and_scale(img_out, (0, 1))

    return img_out


def laplacian_pyramid(gauss_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level

    On each level, the image in Laplacian pyramid is the expansion of the image
    in the lower level of the Gaussian pyramid, subtracted from the Gaussian image
    of the lower level.

    output[k] = gauss_pyr[k] - expand(gauss_pyr[k + 1])

    Note: The last element of output should be identical to the last
    layer of the input pyramid since it cannot be subtracted anymore.

    Args:
        gauss_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
        :param gauss_pyr:
    """
    output = []

    for i in range(len(gauss_pyr) - 1):
        (width, height) = gauss_pyr[i].shape
        output.append(gauss_pyr[i] - expand_image(gauss_pyr[i + 1])[:width, :height])

    output.append(gauss_pyr[len(gauss_pyr) - 1])

    return output


# utility function to plot u, v flow images
def quiver(u, v, scale, stride, color=(0, 255, 0)):
    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):
        for x in range(0, u.shape[1], stride):
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
            if (u[y, x] != 0) & (v[y, x] != 0):
                cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                           y + int(v[y, x] * scale)), (0, 0, 255), 1)

    return img_out


def optic_flow_lk(img_a, img_b, k_size, k_type='uniform', sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    The filter should use a convolution-based method for efficiency, similar
    to the convolution with the kernel in expand_image

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. uniform means a kernel with
                      the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    # assert that input images are in gray scale
    assert len(img_a.shape) < 3, "input image 1 is not in gray scale"
    assert len(img_b.shape) < 3, "input image 2 is not in gray scale"
    #  assert that the input image pixel values are in range [0.0. 1.0]
    assert (img_a >= 0).all() and (img_a <= 1).all(), "input image 1 values are not within range of 0 to 1"
    assert (img_b >= 0).all() and (img_b <= 1).all(), "input image 2 values are not within range of 0 to 1"

    u = []
    v = []
    if k_type == 'uniform':
        kernel_x = np.array([[-1., 1.], [-1., 1.]])
        kernel_y = np.array([[-1., -1.], [1., 1.]])
        kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
        w = k_size / 2  # k_size is odd, all the pixels with offset in between [-w, w] are inside the window
        # Implement Lucas Kanade
        # for each point, calculate I_x, I_y, I_t
        mode = 'same'
        fx = signal.convolve2d(img_a, kernel_x, boundary='symm', mode=mode)
        fy = signal.convolve2d(img_a, kernel_y, boundary='symm', mode=mode)
        ft = signal.convolve2d(img_b, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(img_a, -kernel_t,
                                                                                                boundary='symm',
                                                                                                mode=mode)
        u = np.zeros(img_a.shape)
        v = np.zeros(img_a.shape)
        # within window k_size * k_size
        tau = 0.00000005
        w = int(w)
        for i in range(w, img_a.shape[0] - w):
            for j in range(w, img_a.shape[1] - w):
                ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
                iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
                it = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
                b = np.reshape(it, (it.shape[0], 1))
                a = np.vstack((ix, iy)).T
                # if threshold τ is larger than the smallest eigenvalue of A'A:
                if np.min(abs(np.linalg.eigvals(np.matmul(a.T, a)))) >= tau:
                    nu = np.matmul(np.linalg.pinv(a), b)  # compute the velocity
                    u[i, j] = nu[0]
                    v[i, j] = nu[1]

    elif k_type == 'gaussian':
        kernal = cv2.getGaussianKernel(k_size, sigma)
        w = k_size / 2  # k_size is odd, all the pixels with offset in between [-w, w] are inside the window
        # Implement Lucas Kanade
        # for each point, calculate I_x, I_y, I_t
        mode = 'same'
        fx = signal.convolve2d(img_a, kernal, boundary='symm', mode=mode)
        fy = signal.convolve2d(img_a, kernal, boundary='symm', mode=mode)
        ft = signal.convolve2d(img_b, kernal, boundary='symm', mode=mode) + signal.convolve2d(img_a, kernal,
                                                                                              boundary='symm',
                                                                                              mode=mode)
        u = np.zeros(img_a.shape)
        v = np.zeros(img_a.shape)
        # within window k_size * k_size
        tau = 0.0000000000000000001
        w = int(w)
        for i in range(w, img_a.shape[0] - w):
            for j in range(w, img_a.shape[1] - w):
                ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
                iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
                it = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
                b = np.reshape(it, (it.shape[0], 1))
                a = np.vstack((ix, iy)).T
                if np.min(abs(np.linalg.eigvals(np.matmul(a.T, a)))) >= tau:
                    nu = np.matmul(np.linalg.pinv(a), b)  # compute the velocity
                    u[i, j] = nu[0]
                    v[i, j] = nu[1]

    return u, v


def smooth_image(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def optic_flow_lk_v2(img_a, img_b, k_size, k_type='uniform', sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    The filter should use a convolution-based method for efficiency, similar
    to the convolution with the kernel in expand_image

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. uniform means a kernel with
                      the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    # assert that input images are in gray scale
    assert len(img_a.shape) < 3, "input image 1 is not in gray scale"
    assert len(img_b.shape) < 3, "input image 2 is not in gray scale"
    # assert that the input image pixel values are in range [0.0. 1.0]
    assert (img_a >= 0).all() and (img_a <= 1).all(), "input image 1 values are not within range of 0 to 1"
    assert (img_b >= 0).all() and (img_b <= 1).all(), "input image 2 values are not within range of 0 to 1"

    u = []
    v = []
    if k_type == 'uniform':
        w = k_size / 2  # k_size is odd, all the pixels with offset in between [-w, w] are inside the window
        # Implement Lucas Kanade
        # for each point, calculate I_x, I_y, I_t
        # mode = 'same'
        fx = cv2.Sobel(img_a, cv2.CV_64F, 1, 0, ksize=5)
        fy = cv2.Sobel(img_a, cv2.CV_64F, 0, 1, ksize=5)
        fx_b = cv2.Sobel(img_b, cv2.CV_64F, 1, 0, ksize=5)
        fy_b = cv2.Sobel(img_b, cv2.CV_64F, 0, 1, ksize=5)
        ft = (smooth_image(fy_b) - smooth_image(fx_b)) - (smooth_image(fy) - smooth_image(fx))
        u = np.zeros(img_a.shape)
        v = np.zeros(img_a.shape)
        # within window k_size * k_size
        tau = 0.00000005
        w = int(w)
        for i in range(w, img_a.shape[0] - w):
            for j in range(w, img_a.shape[1] - w):
                ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
                iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
                it = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
                b = np.reshape(it, (it.shape[0], 1))
                a = np.vstack((ix, iy)).T
                # if threshold τ is larger than the smallest eigenvalue of A'A:
                if np.min(abs(np.linalg.eigvals(np.matmul(a.T, a)))) >= tau:
                    nu = np.matmul(np.linalg.pinv(a), b)  # compute the velocity
                    u[i, j] = nu[0]
                    v[i, j] = nu[1]

    elif k_type == 'gaussian':
        kernal = cv2.getGaussianKernel(k_size, sigma)
        w = k_size / 2  # k_size is odd, all the pixels with offset in between [-w, w] are inside the window
        # Implement Lucas Kanade
        # for each point, calculate I_x, I_y, I_t
        mode = 'same'
        fx = signal.convolve2d(img_a, kernal, boundary='symm', mode=mode)
        fy = signal.convolve2d(img_a, kernal, boundary='symm', mode=mode)
        ft = signal.convolve2d(img_b, kernal, boundary='symm', mode=mode) + signal.convolve2d(img_a, kernal,
                                                                                              boundary='symm',
                                                                                              mode=mode)
        u = np.zeros(img_a.shape)
        v = np.zeros(img_a.shape)
        # within window k_size * k_size
        tau = 0.0000000000000000001
        w = int(w)
        for i in range(w, img_a.shape[0] - w):
            for j in range(w, img_a.shape[1] - w):
                ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
                iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
                it = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
                b = np.reshape(it, (it.shape[0], 1))
                a = np.vstack((ix, iy)).T
                if np.min(abs(np.linalg.eigvals(np.matmul(a.T, a)))) >= tau:
                    nu = np.matmul(np.linalg.pinv(a), b)  # compute the velocity
                    u[i, j] = nu[0]
                    v[i, j] = nu[1]

    return u, v


def warp(image, u, v):
    """Warps image using the X and Y displacements (U and V).

    Use cv2.remap to interpolate the warped (dest)image
    and fill the empty pixels from interpolated values of src image

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        u (numpy.array): displacement (in pixels) along X-axis.
        v (numpy.array): displacement (in pixels) along Y-axis.


    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    assert len(image.shape) < 3, "input image is not in gray scale"
    assert (image >= 0).all() and (image <= 1).all(), "input image values are not within range of 0 to 1"

    img = image.copy()
    h, w = img.shape
    # creating a matrix of new coordinate values to which the image will be warped.
    # This produces X and Y such that (𝑋(𝑥, 𝑦), 𝑌(𝑥, 𝑦)) = (𝑥, 𝑦).
    mesh_x, mesh_y = np.meshgrid(range(w), range(h))

    # TODO: print X and Y to verify that (𝑋(𝑥, 𝑦), 𝑌(𝑥, 𝑦)) = (𝑥, 𝑦) for each (x,y) of image
    for x in range(w):
        for y in range(h):
            if (x, y) != (mesh_x[x, y], mesh_y[x, y]):
                print("wrong mesh grid in warp func")
                return

    # TODO: Now add displacements matrices (𝑈, 𝑉) directly with (𝑋, 𝑌) to get the resulting warped locations
    warped = np.zero_like(image)
    for x in range(w):
        for y in range(h):
            warped[y, x] = img[y + v[y, x], x + u[y, x]]

    # TODO: now use cv2.remap to interpolate warped from img and fill empty locations
    # warped_remapped = cv2.remap(img, warped, None, cv2.INTER_CUBIC).transpose(1, 0, 2)

    #  TODO tune cv2.remap interpolation type and border mode to produce best warped image


def scale_u_and_v():
    """Scales up U and V arrays to match the image dimensions assigned
    to the first pyramid level: pyr[0].

    This function takes the U and V arrays computed from an image of a Gaussian
    pyramid level indicated by arg "level" that is smaller than pyr[0],
    and expands them to match a the size of pyr[0].

    This function consists of a sequence of expand_image operations
    based on the pyramid level used to obtain both U and V. Multiply
    the result of expand_image by 2 to scale the vector values. After
    each expand_image operation you should adjust the resulting arrays
    to match the current level shape
    i.e. U.shape == pyr[current_level].shape and
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to
                             pyr[0].shape
    """

    # TODO:

    # TODO create a for loop from level-1 to 0 inclusive to perform the sequence explained above

    raise NotImplementedError


def test_3a_1():
    yos_img_01 = cv2.imread('/DataSeq1/yos_img_01.jpg', 0) / 255.
    yos_img_02 = cv2.imread('/DataSeq1/yos_img_02.jpg', 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) > 0
    k_size = 5  # Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = optic_flow_lk(yos_img_01_g_pyr[level_id],
                         yos_img_02_g_pyr[level_id],
                         k_size, k_type, sigma)

    # u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    # interpolation = cv2.INTER_CUBIC  # You may try different values
    # border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = warp(yos_img_02, u, v)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite("ps4-3-a-1.png", normalize_and_scale(diff_yos_img_01_02))


def test_3a_2():
    yos_img_02 = cv2.imread('yos_img_02.jpg', 0) / 255.
    yos_img_03 = cv2.imread('yos_img_03.jpg', 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = gaussian_pyramid(yos_img_03, levels)

    level_id = 1  # TODO: Select the level number (or id) you wish to use
    k_size = 5  # Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = optic_flow_lk(yos_img_02_g_pyr[level_id],
                         yos_img_03_g_pyr[level_id],
                         k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    # interpolation = cv2.INTER_CUBIC  # You may try different values
    # border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = warp(yos_img_03, u, v)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite("ps4-3-a-2.png", normalize_and_scale(diff_yos_img))


if __name__ == '__main__':
    test_3a_1()
