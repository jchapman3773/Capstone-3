from imageai.Detection import ObjectDetection
from PIL import Image
import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np


def object_detector(filenames,filepaths):
    results_df = pd.DataFrame()
    images = {}

    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "models/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    custom_objects = detector.CustomObjects(person=True, banana=True)
    for f,fp in zip(filenames,filepaths):


        try:
            detections = detector.detectCustomObjectsFromImage(
                                    input_image=fp,
                                    output_type='array',
                                    custom_objects=custom_objects,
                                    minimum_percentage_probability=10,
                                    extract_detected_objects=True)
        except:
            print(f'Error! Image {f} could not be used!')
            continue

        df = pd.DataFrame(detections[1])
        images[f] = Image.fromarray(detections[0])
        
        if 'banana' not in df.name.values:
            print(f'Error! No banana found in image {f}!')
            continue
        if 'person' not in df.name.values:
            print(f'Error! No person found in image {f}!')
            continue
        else:
            df['image'] = np.asarray(detections[2])
            df = df.sort_values(by=['percentage_probability'],ascending=False).groupby('name').head(1)
            dict = {'filename':f}
            dict['image_x'] = detections[0].shape[0]
            dict['image_y'] = detections[0].shape[1]
            for row in df.values:
                dict[f'{row[1]}_box_point1'] = row[0][0]
                dict[f'{row[1]}_box_point2'] = row[0][1]
                dict[f'{row[1]}_box_point3'] = row[0][2]
                dict[f'{row[1]}_box_point4'] = row[0][3]
                dict[f'{row[1]}_pred'] = row[2]
                dict[f'{row[1]}_x'] = row[3].shape[0]
                dict[f'{row[1]}_y'] = row[3].shape[1]
            results_df = results_df.append(dict,ignore_index=True)
            print(f'{f} Predicted')

    return images, results_df

if __name__ == '__main__':
    dir = '/home/julia/Documents/Galvanize/Capstone-3/data/uploads'
    filenames = os.listdir(dir)
    filepaths = []
    for f in filenames:
        filepaths.append(os.path.join(dir,f))

    images, results_df = object_detector(filenames,filepaths)

    for key,value in images.items():
        value.save(f'../data/predictions/{key}')

    engine = create_engine('postgresql://banana:forscale@bananaforscale.ckaldwfguyw5.us-east-2.rds.amazonaws.com:5432/bananaforscale')
    results_df.to_sql('regression',con=engine,if_exists='replace')
