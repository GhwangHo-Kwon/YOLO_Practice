import cv2

# Video, Image path / parameter
video_dir = "./YOLO_V8/Sample_data/"
image_dir = "./YOLO_V8/Sample_data/train/images"
video_name = "robot_fish.mp4"
frame_gap = 30      
video_path = video_dir + '\\' + video_name     


def main():
    # Video open, information
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video size = ({},{}) ".format(width, height))
    
    count = 1                # Image count
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == False:
            print("Can't receive frame")
            break

        # 이미지 추출과 저장
        if(int(cap.get(1)) % frame_gap == 0):
            name =  video_name[:-4] + '_' + str(int(count)) + '.png'   # save frame as png file
            image_path = image_dir + '\\' + name
            cv2.imwrite(image_path, image)   
            print("time :", int(cap.get(1))//3600, "m " ,(int(cap.get(1))%(3600))//60, "s, Saved " + name)
            count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()