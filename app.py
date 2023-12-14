import cv2
from super_gradients.training import models
from super_gradients.common.object_names import Models
from util import find_latest_checkpoint_with_run_id
import torch
import os
import glob

dataset_params = {
    'data_dir':'/content/Multiple-Capuchins-1',
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    'classes': ['Pedro', 'Pele', 'Pery', 'Pesto', 'Pio', 'Pirulo']
}

colors = {
        'Pedro': (255, 0, 0),   # Red
        'Pele': (0, 255, 0),    # Green
        'Pery': (0, 0, 255),    # Blue
        'Pesto': (255, 255, 0), # Cyan
        'Pio': (255, 0, 255),   # Magenta
        'Pirulo': (0, 255, 255) # Yellow
    }

# Load the pre-trained model
new_model = 'multi_capuchin_yolonas_s_run_60epochs'
new_weights = find_latest_checkpoint_with_run_id("weights/"+new_model)
best_model = models.get('yolo_nas_s',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path=new_weights)
model = best_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to process and save the detected frame
def process_frame(frame, frame_index, model, base_output_folder, confidence_threshold=0.7):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = model.predict(frame_rgb)
    
    for frame_prediction in predictions:
        labels = frame_prediction.prediction.labels
        confidence_scores = frame_prediction.prediction.confidence
        bboxes = frame_prediction.prediction.bboxes_xyxy

        for label, conf, bbox in zip(labels, confidence_scores, bboxes):
            if conf >= confidence_threshold:
                label = int(label)
                label_name = dataset_params['classes'][label]

                label_output_folder = os.path.join(base_output_folder, label_name)
                if not os.path.exists(label_output_folder):
                    os.makedirs(label_output_folder)

                # Use the color mapping for the label
                color = colors.get(label_name, (255, 255, 255)) # Default color is white

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, f"{label_name}: {conf:.2f}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                frame_name = f"{label_output_folder}/frame_{frame_index}.jpg"
                cv2.imwrite(frame_name, frame)


def process_webcam(model):
    cap = cv2.VideoCapture(1)
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame, frame_index, model, "output/webcam")
        frame_index += 1
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def process_images_folder(folder, model):
    for frame_index, img_path in enumerate(glob.glob(f"{folder}/*.jpg")):
        frame = cv2.imread(img_path)
        process_frame(frame, frame_index, model, "output/images")


def process_videos_folder(folder, model):
    for video_path in glob.glob(f"{folder}/*.mp4"):
        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame, frame_index, model, "output/videos")
            frame_index += 1
        cap.release()


def main(mode, folder=None):
    if mode == 'webcam':
        process_webcam(model)
    elif mode == 'images':
        process_images_folder(folder, model)
    elif mode == 'videos':
        process_videos_folder(folder, model)
    else:
        print("Invalid mode")

if __name__ == "__main__":
    # Example usage
    main('images', 'images/')
    main('webcam')
    # main('videos', '/path/to/video/folder')