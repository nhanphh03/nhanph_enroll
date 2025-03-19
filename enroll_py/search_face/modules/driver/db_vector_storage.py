from typing import Optional

import numpy as np

from app import db
from modules.driver.vector_storage import VectorStorage


class Vector(db.Model):
    __tablename__ = 'vector'

    id = db.Column(db.String(64), primary_key=True)
    vector = db.Column(db.Binary)


class DbVectorStorage(VectorStorage):
    def __init__(self, db_connection=None):
        if db_connection:
            self.db = db_connection
        else:
            self.db = db

    def add_vector(self, id: str, vector: np.ndarray) -> bool:
        vector_binary = vector.tostring()
        obj = Vector(id=id, vector=vector_binary)
        self.db.session.add(obj)
        self.db.session.commit()
        return True

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        obj = Vector.query.filter(Vector.id == id).first()
        if obj:
            vector_numpy = np.fromstring(obj.vector)
            return vector_numpy
        return None

    def remove_vector(self, id: str) -> bool:
        obj = Vector.query.filter(Vector.id == id).first()
        if obj:
            self.db.session.delete(obj)
            self.db.session.commit()
            return True
        return False
