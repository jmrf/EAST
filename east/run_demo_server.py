#!/usr/bin/env python3
import argparse
import os
import cv2
import uuid
import json
import logging
import numpy as np

from flask import (Flask, request, render_template)

from east.model import EASTPredictor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Config:
    SAVE_DIR = 'east/static/results'


config = Config()


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def save_result(img, rst):
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    return rst


checkpoint_path = './east_icdar2015_resnet_v1_50_rbox'


@app.route('/', methods=['POST'])
def index_post():
    global predictor
    import io
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    rst = predictor.predict(img)

    save_result(img, rst)
    return render_template('index.html', session_id=rst['session_id'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--checkpoint_path', default=checkpoint_path)
    args = parser.parse_args()
    return args


def main():

    # here to avoid issues with argparse adn tf.app.flags
    import tensorflow as tf
    tf.app.flags.DEFINE_string('checkpoint_path', '', '')
    tf.app.flags.DEFINE_string('port', '', '')

    args = get_args()

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))

    global predictor
    predictor = EASTPredictor()
    predictor.load(args.checkpoint_path)

    app.debug = False  # change this to True if you want to debug
    app.run('0.0.0.0', args.port)


if __name__ == '__main__':
    main()
