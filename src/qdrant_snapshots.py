"""
Управление снапшотами Qdrant — создание, восстановление, удаление.
"""
import requests
from typing import Optional

QDRANT_URL = "http://localhost:6333"


def get_collections() -> list[str]:
    """Список всех коллекций."""
    try:
        r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        if r.status_code == 200:
            return [c["name"] for c in r.json()["result"]["collections"]]
    except Exception:
        pass
    return []


def create_snapshot(collection_name: str) -> dict:
    """Создать снапшот коллекции. Возвращает {'name': ..., 'status': ...}."""
    url = f"{QDRANT_URL}/collections/{collection_name}/snapshots"
    r = requests.post(url, timeout=120)
    if r.status_code == 200:
        return r.json().get("result", {})
    return {"error": f"{r.status_code}: {r.text[:200]}"}


def list_snapshots(collection_name: str) -> list[dict]:
    """Список снапшотов коллекции."""
    url = f"{QDRANT_URL}/collections/{collection_name}/snapshots"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        return r.json().get("result", [])
    return []


def restore_from_snapshot(collection_name: str, snapshot_name: str) -> dict:
    """Восстановить коллекцию из снапшота."""
    url = f"{QDRANT_URL}/collections/{collection_name}/snapshots/{snapshot_name}/recover"
    r = requests.put(url, timeout=300)
    if r.status_code == 200:
        return r.json().get("result", {})
    return {"error": f"{r.status_code}: {r.text[:300]}"}


def delete_snapshot(collection_name: str, snapshot_name: str) -> dict:
    """Удалить снапшот."""
    url = f"{QDRANT_URL}/collections/{collection_name}/snapshots/{snapshot_name}"
    r = requests.delete(url, timeout=30)
    if r.status_code == 200:
        return {"status": "deleted"}
    return {"error": f"{r.status_code}: {r.text[:200]}"}


def check_qdrant_health() -> dict:
    """Проверка здоровья Qdrant."""
    try:
        r = requests.get(f"{QDRANT_URL}/healthz", timeout=5)
        return {"status": "ok" if r.status_code == 200 else "error",
                "code": r.status_code}
    except requests.ConnectionError:
        return {"status": "unreachable", "error": "Connection refused"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_collection_info(collection_name: str) -> Optional[dict]:
    """Информация о коллекции (кол-во точек, статус, размерность)."""
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{collection_name}", timeout=10)
        if r.status_code == 200:
            return r.json().get("result", {})
    except Exception:
        pass
    return None


def snapshot_all_collections() -> list[dict]:
    """Создать снапшоты ВСЕХ коллекций. Возвращает список результатов."""
    results = []
    for coll in get_collections():
        res = create_snapshot(coll)
        results.append({"collection": coll, **res})
    return results