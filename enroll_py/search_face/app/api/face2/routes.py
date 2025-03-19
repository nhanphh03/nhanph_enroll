import concurrent.futures
import logging
import os
import uuid

import werkzeug
from PIL import UnidentifiedImageError
from elasticsearch import TransportError
from flask import Blueprint
from flask_restful import Api, Resource, reqparse

from app.api import ResponseStatus
from app.utils import get_image, get_embedding_from_model
from config_run import Config
from modules.driver import elasticsearch_driver
from modules.driver.es_face_storage import EsFaceStorageFactory
from modules.driver.es_vector_storage import EsVectorStorageFactory
from modules.driver.face_storage import FaceStorage
from modules.driver.vector_storage import VectorStorage
from modules.face import embed_face, NoFaceDetection, check_action, check_liveness, get_similars, Command, is_fake, \
    check_fake
from modules.face.liveness import command
from modules.utils import compute_sim

blueprint = Blueprint(
    name='face2',
    import_name=__name__,
    url_prefix='',
)

api = Api(blueprint, prefix='')

LOGGER = logging.getLogger("api")

db_vector: VectorStorage
face_storage: FaceStorage


@blueprint.before_app_first_request
def initialize_api():
    global db_vector, face_storage
    db_vector = EsVectorStorageFactory.get_instance()
    face_storage = EsFaceStorageFactory.get_instance()


class ImageUploadApi(Resource):
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1,
                                                              thread_name_prefix="face search")

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, location=['from', 'args', 'files', 'json'], action='append')
        args_ = parser.parse_args()
        if len(args_['image']) > 0 and "FileStorage" in args_['image'][0]:
            parser.replace_argument('image', type=werkzeug.datastructures.FileStorage, required=True,
                                    location='files',
                                    action='append')

        args = parser.parse_args(strict=True)

        results = []
        for input_data in args.image:
            results.append(self.get_vector_and_save(input_data))
        return {
            'status': ResponseStatus.SUCCESS.value,
            'message': '',
            'data': results,
            'model_name': 'ResNet'
        }, 200

    @staticmethod
    def get_vector_and_save(img_raw):
        file_id = str(uuid.uuid4())
        try:
            img = get_image(img_raw)
            faces = embed_face(img, max_faces=1, fast=False)
            face_emb = get_embedding_from_model(faces)

            if db_vector.add_vector(file_id, face_emb):
                return {
                    'file_id': file_id,
                    'status': ResponseStatus.SUCCESS.value,
                    'message': 'Added vector'}
            else:
                return {
                    'file_id': file_id,
                    'status': ResponseStatus.INPUT_ERROR.value,
                    'message': f'{file_id} has exists.'
                }
        except NoFaceDetection as e:
            LOGGER.exception(e)
            return {
                'file_id': None,
                'status': ResponseStatus.NO_FACE_ERROR.value,
                'message': str(e)
            }
        except Exception as e:
            LOGGER.exception(e)
            return {
                'file_id': None,
                'status': ResponseStatus.INPUT_ERROR.value,
                'message': 'Image input was wrong format. Please check your input image.'
            }


class LastFileIdApi(Resource):
    def post(self):
        return {
            'status': ResponseStatus.SUCCESS.value,
            'last_ids': db_vector.get_last_id(5),
            'model_name': 'ResNet'
        }, 200


class FaceCompareApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('file_id_1', required=True, type=str, location=['json'],
                            help='Required id of image ({error_msg})')
        parser.add_argument('file_id_2', required=True, type=str, location=['json'],
                            help='Required id of image ({error_msg})')
        args = parser.parse_args()
        file_id_1 = args.get('file_id_1')
        file_id_2 = args.get('file_id_2')

        vector1 = db_vector.get_vector(file_id_1)
        if vector1 is None:
            return {
                'status': ResponseStatus.INPUT_ERROR.value,
                'message': f"Not found file_id_1='{file_id_1}'.",
                'compare_result': None,
                'similar_score': None,
                'model_name': 'ResNet'
            }, 400
        vector2 = db_vector.get_vector(file_id_2)
        if vector2 is None:
            return {
                'status': ResponseStatus.INPUT_ERROR.value,
                'message': f"Not found file_id_2='{file_id_2}'.",
                'compare_result': None,
                'similar_score': None,
                'model_name': 'ResNet'
            }, 400
        sim = compute_sim(vector1, vector2, new_range=True)
        match = sim >= Config.similar_thresh_compare
        return {
            'status': ResponseStatus.SUCCESS.value,
            'message': '',
            "compare_result": "MATCH" if match else "DO_NOT_MATCH",
            "similar_score": sim,
            'model_name': 'ResNet'
        }, 200


class FaceSearchApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('file_id', required=True, type=str, location=['json'],
                            help='Required id of image ({error_msg})')
        parser.add_argument('source', required=True, type=str, location=['json'],
                            help='Source of face. ({error_msg})')
        args = parser.parse_args()
        file_id = args.get('file_id')
        source = args.get('source')

        vector = db_vector.get_vector(file_id)
        if vector is None:
            return {
                'status': ResponseStatus.INPUT_ERROR.value,
                'message': f"Not found file_id='{file_id}'.",
                'data': [],
                'model_name': 'ResNet'
            }, 400
        try:
            res_es = face_storage.search(people_id=None,
                                         face=vector,
                                         source=source,
                                         meta_data=None)
            return {
                'status': ResponseStatus.SUCCESS.value,
                'message': f'Found {len(res_es)} records.',
                'data': res_es,
                'model_name': 'ResNet'
            }, 200
        except TransportError as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.ERROR.value,
                'message': "Can not connect to database.",
                'data': [],
                'model_name': 'ResNet'
            }, 500
        except Exception as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.ERROR.value,
                'message': 'Internal server error',
                'data': [],
                'model_name': 'ResNet'
            }, 500


class FaceRegisterApi(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('file_id', type=str, required=True, location=['json'])
        parser.add_argument('people_id', type=str, required=True, location=['json'])
        parser.add_argument('source', type=str, default='default', location=['json'])
        parser.add_argument('meta_data', type=dict, required=True, location=['json'])
        args = parser.parse_args()
        people_id = args.get('people_id')
        source = args.get('source')
        file_id = args.get('file_id')
        meta_data = args.get('meta_data')
        vector = db_vector.get_vector(file_id)
        if vector is None:
            return {
                'status': ResponseStatus.INPUT_ERROR.value,
                'message': f'Not found file_id={file_id}.',
                'model_name': 'ResNet'
            }

        try:
            es_res = face_storage.insert(file_id=file_id,
                                         people_id=people_id,
                                         face=vector,
                                         source=source,
                                         meta_data=meta_data)
            es_res['model_name']='ResNet'
            return es_res, 200

        except Exception as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.ERROR.value,
                'message': 'Internal server error',
                'model_name': 'ResNet'
            }, 500


class FaceRegisterApiMulti(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('file_id', type=str, required=True, location=['json'], action='append')
        parser.add_argument('people_id', type=str, required=True, location=['json'])
        parser.add_argument('source', type=str, default='default', location=['json'])
        parser.add_argument('meta_data', type=dict, required=True, location=['json'])
        args = parser.parse_args()
        people_id = args.get('people_id')
        source = args.get('source')
        file_ids = args.get('file_id')
        meta_data = args.get('meta_data')
        vectors = [db_vector.get_vector(file_id) for file_id in file_ids]
        response = []
        for file_id, vector in zip(file_ids, vectors):
            if vector is None:
                response.append({
                    'file_id': file_id,
                    'status': ResponseStatus.INPUT_ERROR.value,
                    'message': f'Not found file_id={file_id}.'
                })

            try:
                es_res = face_storage.insert(file_id=file_id,
                                             people_id=people_id,
                                             face=vector,
                                             source=source,
                                             meta_data=meta_data)
                es_res['file_id'] = file_id
                response.append(es_res)

            except Exception as e:
                LOGGER.exception(e)
                response.append({
                    'file_id': file_id,
                    'status': ResponseStatus.ERROR.value,
                    'message': 'Internal server error'
                })
        return {
            'status': ResponseStatus.SUCCESS.value,
            'message': '',
            'data': response,
            'model_name': 'ResNet'
        }, 200


class UpdateMetaDataApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('people_id', type=str, required=True, location=['json'])
        parser.add_argument('source', type=str, default='default', location=['json'])
        parser.add_argument('meta_data', type=dict, required=True, location=['json'])
        parser.add_argument('file_id', type=str, required=False, default=None, location=['json'])
        args = parser.parse_args()
        people_id = args.get('people_id')
        source = args.get('source')
        file_id = args.get('file_id')
        meta_data = args.get('meta_data')
        count = face_storage.update_metadata(people_id, source, meta_data, file_id)
        type_search = 'people' if file_id is None else 'file_id'
        key_search = f"'{people_id}' at '{source}'" if file_id is None else f"'{file_id}'"
        if count > 0:
            return {
                'status': ResponseStatus.SUCCESS.value,
                'message': f"{count} records of {type_search} {key_search} has updated meta data",
                'model_name': 'ResNet'
            }, 200
        return {
            'status': ResponseStatus.INPUT_ERROR.value,
            'message': f"Not found {type_search} {key_search} to update",
            'model_name': 'ResNet'
        }, 400


class DeletePeopleApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('people_id', type=str, required=True, location=['json'])
        parser.add_argument('source', type=str, default='default', location=['json'])
        parser.add_argument('file_id', type=str, default=None, location=['json'])
        args = parser.parse_args()
        people_id = args.get('people_id')
        source = args.get('source')
        file_id = args.get('file_id')
        count = face_storage.delete(people_id,
                                    source,
                                    file_id)
        if count:
            return {
                'status': ResponseStatus.SUCCESS.value,
                'message': f"{count} records of people '{people_id}' at '{source}' has deleted",
                'model_name': 'ResNet'
            }, 200
        return {
            'status': ResponseStatus.INPUT_ERROR.value,
            'message': f"Not found people '{people_id}' at '{source}' to delete",
            'model_name': 'ResNet'
        }, 400


class FaceLiveCheckApi(Resource):
    def __init__(self):
        pass

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, location=['json'])
        parser.add_argument('cmds', type=dict, location=['json'], action='append')
        args = parser.parse_args()

        try:
            image = get_image(args.image)
            cmds = args.cmds
            for cmd in cmds:
                cmd['action'] = command.get_command(cmd['action'])
                for i, img in enumerate(cmd['images']):
                    cmd['images'][i] = get_image(img)

            main_faces = embed_face(image, max_faces=1, fast=True)
            main_face_emb = get_embedding_from_model(main_faces)

            action_results = []
            # Get embedding and compare
            for cmd in cmds:
                embs = [get_embedding_from_model(embed_face(img)) for img in cmd['images']]
                matches, sims = get_similars(main_face_emb, embs, Config.similar_thresh_compare)
                is_liveness = False
                if len(matches) > 0:
                    is_liveness = check_liveness(matches, sims)

                result_check_action = None
                if cmd['action'] == Command.PORTRAIT:
                    cmd['check_anti_spoof'] = True
                else:
                    result_check_action = check_action(cmd['images'], cmd['action'])

                result_check_anti_spoof = None
                if cmd.get('check_anti_spoof', False):
                    fake_probs = []
                    for img in cmd['images']:
                        fake_probs.append(is_fake(img)['fake_prob'][0])
                    fake_prob = sum(fake_probs) / len(fake_probs)
                    result_check_anti_spoof = fake_prob

                action_results.append({
                    'cmd': cmd['action'].name,
                    'pass_action': result_check_action,
                    'similar_score': [round(s * 100, 5) for s in sims],
                    'similar': is_liveness,
                    'is_fake': result_check_anti_spoof > Config.face_anti_spoof_threshold if result_check_anti_spoof else None,
                    'fake_score': round(result_check_anti_spoof * 100, 5) if result_check_anti_spoof else None
                })

            return {
                "status": ResponseStatus.SUCCESS.value,
                "message": "",
                "data": action_results,
                'model_name': 'ResNet'
            }, 200
        except UnidentifiedImageError as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.INPUT_ERROR.value,
                'message': 'Wrong image input format.',
                'data': [],
                'model_name': 'ResNet'
            }, 400

        except NoFaceDetection as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.NO_FACE_ERROR.value,
                'message': str(e),
                'data': [],
                'model_name': 'ResNet'
            }, 200

        except Exception as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.ERROR.value,
                'message': 'Internal server error',
                'data': [],
                'model_name': 'ResNet'
            }, 500


class FaceAntiSpoofApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, location=['from', 'args', 'files', 'json'], action='append')
        args_ = parser.parse_args()
        if len(args_['image']) > 0 and "FileStorage" in args_['image'][0]:
            parser.replace_argument('image', type=werkzeug.datastructures.FileStorage, required=True,
                                    location='files',
                                    action='append')

        args = parser.parse_args(strict=True)
        try:
            images = [get_image(img_raw) for img_raw in args.image]
            res = check_fake(images, version=4)
            results = []
            for pred, score in zip(res['is_fake'], res['fake_prob']):
                results.append({
                    'status': ResponseStatus.SUCCESS.value,
                    'message': '',
                    "fake_prob": round(score * 100, 5),
                    "is_fake": pred
                })

            return {
                'status': ResponseStatus.SUCCESS.value,
                'message': '',
                'data': results,
                'model_name': 'ResNet'
            }, 200
        except UnidentifiedImageError as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.SUCCESS.value,
                'message': 'Wrong image input format.',
                'data': None,
                'model_name': 'ResNet'
            }, 400
        except Exception as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.SUCCESS.value,
                'message': 'Internal Server Error',
                'data': None,
                'model_name': 'ResNet'
            }, 500


api.add_resource(ImageUploadApi, '/api/v1/upload')
api.add_resource(FaceCompareApi, '/api/v3/face-compare')
api.add_resource(FaceSearchApi, '/api/v3/face-search')
api.add_resource(FaceRegisterApi, '/api/v3/face-register')
api.add_resource(FaceRegisterApiMulti, '/api/v3/face-register-multi')
api.add_resource(UpdateMetaDataApi, '/api/v1/meta-data-update')
api.add_resource(DeletePeopleApi, '/api/v1/delete')
api.add_resource(LastFileIdApi, '/api/v1/last-id')
api.add_resource(FaceLiveCheckApi, '/api/v2/live-check')
api.add_resource(FaceAntiSpoofApi, '/api/v3/anti-spoof')
