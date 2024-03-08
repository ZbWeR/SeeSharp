# 读取 samples 下的图片文件, 利用 cv 进行二维码识别
import cv2
from pyzbar import pyzbar

def decode_qr_code(image):
    # 读取图片
    img = cv2.imread(image)
    # 识别图片中所有的二维码
    barcode_list = pyzbar.decode(img)
    
    for barcode in barcode_list:
        # 提取二维码的位置信息, 并绘制矩形框以突出显示
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 提取二维码的数据信息与类型
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # 文本居中显示
        info = f"{barcodeData}"
        info_size,_ = cv2.getTextSize(info, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        text_x = x + w // 2 - info_size[0] // 2
        text_y = y + h // 2 - info_size[1] // 2
        cv2.putText(img, info, (text_x,text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

        print(f"[INFO] Found {barcodeType} barcode: {barcodeData}")

    # 显示图片并等待按键
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

decode_qr_code('samples/qr2.jpg')