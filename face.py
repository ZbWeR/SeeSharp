import cv2
from deepface import DeepFace

# 图像文件路径
img_path = './samples/emotion2.jpg'

try:
    # 使用DeepFace进行人脸表情识别
    result = DeepFace.analyze(img_path, actions=['emotion'])

    # 获取表情和置信度
    emotion = result[0]['dominant_emotion']
    confidence = result[0]['emotion'][emotion]

    # 打印表情和置信度
    print(f"Dominant Emotion: {emotion}, Confidence: {confidence:.2f}%")

    # 读取图像
    frame = cv2.imread(img_path)

    # 在图像上显示表情
    cv2.putText(frame, f"{emotion} ({confidence:.2f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示图像
    cv2.namedWindow('Emotion Recognition', cv2.WINDOW_NORMAL)
    cv2.imshow('Emotion Recognition', frame)
    cv2.waitKey(0)  # 等待按键
except Exception as e:
    print("Error:", e)

cv2.destroyAllWindows()
