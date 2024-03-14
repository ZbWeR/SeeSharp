import cv2
from pyzbar.pyzbar import decode
import numpy as np


def decode_qrcode(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("无法读取图像: {}".format(image_path))

        decoded_objects = decode(image)
        if not decoded_objects:
            raise ValueError("图像中未找到二维码: {}".format(image_path))

        return decoded_objects, image
    except Exception as e:
        print("二维码解码错误: {}".format(e))
        return None, None


def cover_qrcode_with_image(original_image, decoded_objects, cover_image_path):
    try:
        cover_image = cv2.imread(cover_image_path)
        if cover_image is None:
            raise FileNotFoundError("无法读取覆盖图像: {}".format(cover_image_path))

        for obj in decoded_objects:
            points = obj.polygon
            # 如果识别到的二维码点数大于4，则使用凸包算法获取四个点
            if len(points) > 4:
                hull = cv2.convexHull(
                    np.array([point for point in points], dtype=np.float32))
                points = hull

            # 获取二维码四个点坐标
            qr_points = np.array([[pt.x, pt.y] for pt in points],
                                 dtype=np.float32)
            qr_points = qr_points[:4]

            # 构造透视变换矩阵
            cover_points = np.array(
                [[0, 0], [cover_image.shape[1], 0],
                 [cover_image.shape[1], cover_image.shape[0]],
                 [0, cover_image.shape[0]]],
                dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(cover_points, qr_points)
            transformed_image = cv2.warpPerspective(
                cover_image, matrix,
                (original_image.shape[1], original_image.shape[0]))

            # 创建一个与原始图像大小相同的黑色遮罩
            mask = np.zeros_like(original_image, dtype=np.uint8)
            # 在二维码所在区域填充白色
            cv2.fillConvexPoly(mask, np.int32(qr_points), (255, 255, 255))
            # 按位与，得到一张二维码区域为覆盖图像，其他区域为黑色的图像
            transformed_image = cv2.bitwise_and(transformed_image, mask)
            # 按位与，得到一张二维码区域为黑色，其他区域为原始图像的图像
            original_image = cv2.bitwise_and(original_image,
                                             cv2.bitwise_not(mask))
            # 将二维码区域的覆盖图像与原始图像相加
            original_image = cv2.add(original_image, transformed_image)

        return original_image
    except Exception as e:
        print("覆盖图像错误: {}".format(e))
        return None


def main():
    decoded_objects, image = decode_qrcode("./samples/qr2.jpg")
    if decoded_objects is not None and image is not None:
        result_image = cover_qrcode_with_image(image, decoded_objects,
                                               "./samples/demo.png")
        if result_image is not None:
            cv2.imwrite('result_image.jpg', result_image)
            print("二维码已成功覆盖✔️")
        else:
            print("无法完成二维码的覆盖。")
    else:
        print("未找到有效的二维码或图像。")


main()
