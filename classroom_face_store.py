import os
import pickle


def load_face_database(database_path):
    if os.path.exists(database_path):
        try:
            with open(database_path, "rb") as f:
                data = pickle.load(f)
                encodings = data.get("encodings", [])
                names = data.get("names", [])
            print(f"已加载人脸数据库：{len(names)} 人")
            return encodings, names
        except Exception as e:
            print(f"加载人脸数据库失败：{e}")
            return [], []

    print("人脸数据库不存在，将创建新数据库")
    return [], []


def save_face_database(database_path, encodings, names):
    try:
        data = {
            "encodings": encodings,
            "names": names,
        }
        with open(database_path, "wb") as f:
            pickle.dump(data, f)
        print(f"人脸数据库已保存：{len(names)} 人")
        return True
    except Exception as e:
        print(f"保存人脸数据库失败：{e}")
        return False


def get_face_database_entries(names):
    return [{"index": idx, "name": name} for idx, name in enumerate(names)]


def delete_face_from_database(encodings, names, index, save_func):
    if index < 0 or index >= len(names):
        return False, "无效的人脸索引"

    try:
        deleted_name = names.pop(index)
        encodings.pop(index)
        if not save_func():
            return False, "删除后保存人脸数据库失败"
        print(f"已删除人脸：{deleted_name}")
        return True, deleted_name
    except Exception as e:
        return False, f"删除失败：{e}"


def rename_face_in_database(names, index, new_name, save_func):
    normalized_name = new_name.strip()
    if not normalized_name:
        return False, "姓名不能为空"
    if index < 0 or index >= len(names):
        return False, "无效的人脸索引"

    try:
        old_name = names[index]
        names[index] = normalized_name
        if not save_func():
            names[index] = old_name
            return False, "重命名后保存人脸数据库失败"
        print(f"已重命名人脸：{old_name} -> {normalized_name}")
        return True, old_name
    except Exception as e:
        return False, f"重命名失败：{e}"
