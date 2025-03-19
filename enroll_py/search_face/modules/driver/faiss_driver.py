import time
import faiss
import numpy as np
import psutil
import os
import pickle
import logging

from modules import utils

LOGGER = logging.getLogger('model')

process = psutil.Process(os.getpid())


def get_unique_name_from_face_id(face_id):
    elems = face_id.split('_')
    if len(elems) >= 3:
        unique_name = face_id.replace('_{}_{}'.format(elems[-2], elems[-1]), '')
    else:
        unique_name = face_id
    return unique_name


class FaissSearching:
    def __init__(self, faceid_db):
        """
        index by faiss
        :param faceid_db: the mongodb collection instance contain face feature
        """
        self.faceid_db = faceid_db
        self.total_people, self.faiss_index, self.list_face_idx_with_face_id, self.face_id_info, self.face_id_feat = self.init_faiss_index()

    def init_faiss_index(self):
        s = time.time()
        col = self.faceid_db.find({})
        total_people = col.count()
        print('FAISS for {} people searching'.format(total_people))
        max_face_per_person = 10
        list_face_idx_with_face_id = []
        face_id_info = []
        face_id_feat = []
        total_faces = 0
        d = 512
        features_ar = []
        if total_people > 1000000:
            LOGGER.info("training...")
            nlist = int(np.sqrt(total_people))
            quantizer = faiss.IndexFlatL2(d)  # build the index
            faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            assert not faiss_index.is_trained
            s_train = time.time()
            for doc in col:
                for j in range(len(doc['faces'])):
                    feat = np.array(doc['faces'][j]['feature']).astype('float32')
                    features_ar.append(feat)
                if len(features_ar) > 1000:
                    features_ar = np.asarray(features_ar).astype('float32')
                    faiss_index.train(features_ar)
                    del features_ar
                    features_ar = []
                    LOGGER.info(
                        "Memory while training: {}GB, total faces: {}".format(process.memory_info().rss / 1073741824,
                                                                              total_faces))
            # the last one
            if len(features_ar) > 0:
                features_ar = np.asarray(features_ar).astype('float32')
                faiss_index.train(features_ar)
                del features_ar
                LOGGER.info(
                    "Memory while training: {}GB, total faces: {}".format(process.memory_info().rss / 1073741824,
                                                                          total_faces))
            e_train = time.time()
            LOGGER.info('train elapsed time: {:.2f}ms'.format((e_train - s_train) * 1000))
            faiss_index.nprobe = 5
        else:
            faiss_index = faiss.IndexFlatL2(d)

        LOGGER.info("Memory before index: ", process.memory_info().rss / 1073741824, "GB")  # in GB
        features_ar = []
        col = self.faceid_db.find({})
        for doc in col:
            unique_name = doc['unique_name']
            for j in range(len(doc['faces'][:max_face_per_person])):
                feat = np.array(doc['faces'][j]['feature']).astype('float32')
                features_ar.append(feat)
                face_id = doc['faces'][j]['face_id']
                list_face_idx_with_face_id.append(face_id)
                if doc.__contains__('info'):
                    face_id_info.append(doc['info'])
                else:
                    face_id_info.append({})
                face_id_feat.append(feat)
                total_faces += 1
            if len(features_ar) > 10000:
                features_ar = np.asarray(features_ar).astype('float32')
                faiss_index.add(features_ar)
                del features_ar
                features_ar = []
                LOGGER.info(
                    "Memory while indexing: {}GB, total faces: {}".format(process.memory_info().rss / 1073741824,
                                                                          total_faces))
        # the last one
        if len(features_ar) > 0:
            features_ar = np.asarray(features_ar).astype('float32')
            faiss_index.add(features_ar)
            del features_ar
            LOGGER.info("Memory while indexing: {}GB, total faces: {}".format(process.memory_info().rss / 1073741824,
                                                                              total_faces))

        LOGGER.info(
            "Memory after index {} faces: {}GB".format(faiss_index.ntotal, process.memory_info().rss / 1073741824))
        e = time.time()
        LOGGER.info(
            'Init FAISS for {} people with {} faces, elapsed {}ms'.format(total_people, total_faces, (e - s) * 1000))
        return total_people, faiss_index, list_face_idx_with_face_id, face_id_info, face_id_feat

    def delete_faces_by_id(self, deleted_face_ids):
        """
        delete faceid from faiss index
        :param deleted_face_ids:
        :return:
        """
        for face_id in deleted_face_ids:
            try:
                idx = self.list_face_idx_with_face_id.index(face_id)
                self.faiss_index.remove_ids(np.array([idx]))
                self.list_face_idx_with_face_id = self.list_face_idx_with_face_id[
                                                  :idx] + self.list_face_idx_with_face_id[idx + 1:]
                self.face_id_feat = self.face_id_feat[:idx] + self.face_id_feat[idx + 1:]
                self.face_id_info = self.face_id_info[:idx] + self.face_id_info[idx + 1:]
            except Exception as e:
                LOGGER.exception(e)
                return False
        return True

    def add_new_face_by_id(self, new_feats, new_face_ids, infos):
        """
        Index new face. Update self.total_people, self.faiss_index, self.list_face_idx_with_face_id
        """
        features_ar = new_feats
        try:
            if self.total_people < 1000000:
                self.face_id_feat += new_feats
                self.face_id_info += infos
                new_feats = np.asarray(new_feats).astype('float32')
                self.faiss_index.add(new_feats)
                self.list_face_idx_with_face_id += new_face_ids
            else:
                self.face_id_feat += new_feats
                self.face_id_info += infos
                new_feats = np.asarray(new_feats).astype('float32')
                self.faiss_index.train(new_feats)
                self.list_face_idx_with_face_id += new_face_ids
            return True
        except Exception as e:
            LOGGER.exception(e)
        return False

    def get_top_k_face(self, face_feature, K=5, norm_score=1):
        if len(self.list_face_idx_with_face_id) == 0:
            return []
        face_feature = np.asarray([face_feature]).astype('float32')
        # s = time.time()
        D, I = self.faiss_index.search(face_feature, K)
        # e = time.time()
        # print((e-s)*1000)
        res = []
        for i in range(K):
            idx = I[0][i] % self.faiss_index.ntotal
            if len(self.list_face_idx_with_face_id) <= idx:
                continue
            unique_name_by_face_id = get_unique_name_from_face_id(self.list_face_idx_with_face_id[idx])
            face_id = self.list_face_idx_with_face_id[idx]
            db_face = list(self.faceid_db.find({'unique_name': unique_name_by_face_id}))

            info = self.face_id_info[idx]
            feat = self.face_id_feat[idx]
            if norm_score == 1:
                compare_score = utils.normalC(np.dot(face_feature[0], feat.T))
            else:
                compare_score = np.dot(face_feature[0], feat.T)

            res.append({"unique_name": unique_name_by_face_id, "person_name": unique_name_by_face_id,
                        "info": info,
                        "face_id": face_id, "compare_score": float(compare_score)})
        return res


if __name__ == '__main__':
    import config

    faces_dir = config.face_recognizer_conf.known_face_path
    face_db_host = config.face_recognizer_conf.MONGODB_HOST
    face_db_name = config.face_recognizer_conf.DB_NAME
    known_faces_collection_name = config.face_recognizer_conf.COLLECTION_NAME

    # read data
    known_people_path = os.path.join(faces_dir, '{}_{}.pick'.format(face_db_name, known_faces_collection_name))
    known_people = pickle.load(open(known_people_path, 'rb'))
    known_faces_index = []
    features_ar = []
    fid = 0
    for person_idx, person in enumerate(known_people):
        for i in range(len(person['faces'])):
            db_feature = person['faces'][i]['feature']
            features_ar.append(db_feature)
            fid += 1
            known_faces_index.append(
                {"unique_name": person['unique_name'], "person_name": person["person_name"],
                 "face_id": person['faces'][i]['face_id']})
    features_ar = np.asarray(features_ar).astype('float32')
    print("features_ar: ", features_ar.shape)  # ~500K
    d = features_ar.shape[1]
    nlist = 1000
    quantizer = faiss.IndexFlatL2(d)  # build the index
    faiss_index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)

    assert not faiss_index.is_trained
    s = time.time()
    faiss_index.train(features_ar)
    e = time.time()
    print('train elapsed time: {:.2f}ms'.format((e - s) * 1000))
    assert faiss_index.is_trained

    for i in [0, 1, 2]:
        print(process.memory_info().rss / 2014 / 1024 / 1024, "GB")  # in GB
        faiss_index.add(features_ar)  # add vectors to the index
        print(faiss_index.ntotal)
        print(process.memory_info().rss / 2014 / 1024 / 1024, "GB")  # in GB

    # query
    s = time.time()
    n = len(features_ar)
    face_feature = features_ar[:n]
    D, I = faiss_index.search(face_feature, 5)  # sanity check
    e = time.time()
    print('elapsed time: {:.2f}ms'.format((e - s) * 1000 / n))
