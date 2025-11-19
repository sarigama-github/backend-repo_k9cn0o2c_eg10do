import os
from typing import List, Optional, Tuple, Dict, Set
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import db, create_document, get_documents
from schemas import Gamestate, Stats
import random
import copy

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Sudoku Core -----------------
Grid = List[List[int]]

def find_empty(grid: Grid) -> Optional[Tuple[int, int]]:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None

def is_valid(grid: Grid, r: int, c: int, val: int) -> bool:
    # Row
    if any(grid[r][i] == val for i in range(9)):
        return False
    # Col
    if any(grid[i][c] == val for i in range(9)):
        return False
    # Box
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if grid[i][j] == val:
                return False
    return True

def solve_grid(grid: Grid) -> bool:
    empty = find_empty(grid)
    if not empty:
        return True
    r, c = empty
    nums = list(range(1, 10))
    for val in nums:
        if is_valid(grid, r, c, val):
            grid[r][c] = val
            if solve_grid(grid):
                return True
            grid[r][c] = 0
    return False

def count_solutions(grid: Grid, limit: int = 2) -> int:
    # Backtracking count up to limit
    count = 0
    def backtrack(g: Grid):
        nonlocal count
        if count >= limit:
            return
        empty = find_empty(g)
        if not empty:
            count += 1
            return
        r, c = empty
        for v in range(1, 10):
            if is_valid(g, r, c, v):
                g[r][c] = v
                backtrack(g)
                g[r][c] = 0
                if count >= limit:
                    return
    backtrack([row[:] for row in grid])
    return count

def generate_complete_grid() -> Grid:
    grid = [[0 for _ in range(9)] for _ in range(9)]
    def fill():
        empty = find_empty(grid)
        if not empty:
            return True
        r, c = empty
        nums = list(range(1, 10))
        random.shuffle(nums)
        for v in nums:
            if is_valid(grid, r, c, v):
                grid[r][c] = v
                if fill():
                    return True
                grid[r][c] = 0
        return False
    fill()
    return grid

DIFFICULTY_HINTS = {
    "easy": (38, 40),
    "medium": (30, 33),
    "hard": (25, 28),
}


def make_puzzle(solution: Grid, hints: int) -> Grid:
    # Start from full solution and remove cells symmetrically until desired hints remain
    puzzle = [row[:] for row in solution]
    # positions in one half to enforce symmetry
    positions = [(r, c) for r in range(9) for c in range(9) if (r, c) <= (8 - r, 8 - c)]
    random.shuffle(positions)
    cells_to_remove = 81 - hints
    removed = 0
    for (r, c) in positions:
        if removed >= cells_to_remove:
            break
        # symmetric counterpart
        r2, c2 = 8 - r, 8 - c
        # Temporarily remove
        saved1, saved2 = puzzle[r][c], puzzle[r2][c2]
        if saved1 == 0 and saved2 == 0:
            continue
        puzzle[r][c] = 0
        if (r2, c2) != (r, c):
            puzzle[r2][c2] = 0
        # Check uniqueness
        if count_solutions([row[:] for row in puzzle], limit=2) == 1:
            removed += 2 if (r2, c2) != (r, c) else 1
        else:
            # revert
            puzzle[r][c] = saved1
            if (r2, c2) != (r, c):
                puzzle[r2][c2] = saved2
    return puzzle


def compute_conflicts(grid: Grid, r: int, c: int, val: int) -> Set[Tuple[int, int]]:
    conflicts: Set[Tuple[int, int]] = set()
    if val == 0:
        return conflicts
    # Row/Col
    for i in range(9):
        if grid[r][i] == val and i != c:
            conflicts.add((r, i))
        if grid[i][c] == val and i != r:
            conflicts.add((i, c))
    # Box
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if grid[i][j] == val and (i, j) != (r, c):
                conflicts.add((i, j))
    return conflicts

# --------------- API Models -----------------
class NewGameRequest(BaseModel):
    device_id: str
    difficulty: str

class MoveRequest(BaseModel):
    device_id: str
    row: int
    col: int
    value: int  # 0 to clear, 1-9 to place
    note_mode: bool = False

class EraseRequest(BaseModel):
    device_id: str
    row: int
    col: int

class TimerUpdate(BaseModel):
    device_id: str
    elapsed_seconds: int

# --------------- Helpers -----------------

def get_gamestate(device_id: str) -> Optional[Dict]:
    docs = get_documents("gamestate", {"device_id": device_id}, limit=1)
    return docs[0] if docs else None


# ---------------- Routes ------------------
@app.get("/")
def read_root():
    return {"message": "Sudoku Backend Running"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from Sudoku API"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set"
            response["database_name"] = db.name
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    diff = req.difficulty.lower()
    if diff not in DIFFICULTY_HINTS:
        raise HTTPException(status_code=400, detail="Invalid difficulty")
    hints = random.randint(*DIFFICULTY_HINTS[diff])
    solution = generate_complete_grid()
    puzzle = make_puzzle(solution, hints)
    fixed = [[puzzle[r][c] != 0 for c in range(9)] for r in range(9)]
    current = copy.deepcopy(puzzle)
    notes: List[List[List[int]]] = [[[ ] for _ in range(9)] for _ in range(9)]

    gs = Gamestate(
        device_id=req.device_id,
        difficulty=diff,
        puzzle=puzzle,
        solution=solution,
        current=current,
        fixed=fixed,
        notes=notes,
        mistakes=0,
        elapsed_seconds=0,
        is_completed=False,
    )
    # Upsert behavior: remove existing then insert
    db["gamestate"].delete_many({"device_id": req.device_id})
    create_document("gamestate", gs)

    # Ensure stats exists
    if not get_documents("stats", {"device_id": req.device_id}, limit=1):
        st = Stats(device_id=req.device_id, games_played=0, games_won=0, best_time_seconds=None, last_difficulty=diff, total_mistakes=0)
        create_document("stats", st)
    else:
        db["stats"].update_one({"device_id": req.device_id}, {"$set": {"last_difficulty": diff}}, upsert=True)

    return {"status": "ok", "gamestate": gs.model_dump()}

@app.get("/api/game/{device_id}")
def get_game(device_id: str):
    gs = get_gamestate(device_id)
    if not gs:
        return {"status": "empty"}
    # Convert ObjectId
    gs.pop("_id", None)
    return {"status": "ok", "gamestate": gs}

@app.post("/api/move")
def make_move(req: MoveRequest):
    gs = get_gamestate(req.device_id)
    if not gs:
        raise HTTPException(status_code=404, detail="No game found")
    r, c = req.row, req.col
    if gs["fixed"][r][c]:
        raise HTTPException(status_code=400, detail="Cannot change a fixed cell")

    is_note = req.note_mode
    conflicts: List[Tuple[int, int]] = []

    if is_note:
        # toggle note value in notes list for that cell (1-9 only)
        if req.value < 1 or req.value > 9:
            raise HTTPException(status_code=400, detail="Note value must be 1-9")
        notes: List[int] = gs["notes"][r][c]
        if req.value in notes:
            notes.remove(req.value)
        else:
            notes.append(req.value)
            notes.sort()
        db["gamestate"].update_one({"device_id": req.device_id}, {"$set": {f"notes.{r}.{c}": notes}})
    else:
        # place value or clear
        val = req.value
        current = gs["current"]
        if val == 0:
            current[r][c] = 0
            db["gamestate"].update_one({"device_id": req.device_id}, {"$set": {f"current.{r}.{c}": 0, f"notes.{r}.{c}": []}})
        else:
            # validate conflicts vs row/col/box
            current[r][c] = val
            conflicts_set = compute_conflicts(current, r, c, val)
            conflicts = list(conflicts_set)
            # If conflicts exist or value != solution at that position -> mistake
            mistake = False
            if conflicts_set:
                mistake = True
            elif gs["solution"][r][c] != val:
                mistake = True
            if mistake:
                gs_mistakes = int(gs.get("mistakes", 0)) + 1
                db["gamestate"].update_one({"device_id": req.device_id}, {"$set": {f"current.{r}.{c}": val}, "$inc": {"mistakes": 1}})
                db["stats"].update_one({"device_id": req.device_id}, {"$inc": {"total_mistakes": 1}})
            else:
                # correct move: set value and clear notes in that cell
                db["gamestate"].update_one({"device_id": req.device_id}, {"$set": {f"current.{r}.{c}": val, f"notes.{r}.{c}": []}})

            # Check completion
            updated_gs = get_gamestate(req.device_id)
            done = updated_gs and updated_gs["current"] == updated_gs["solution"]
            if done and not updated_gs.get("is_completed"):
                db["gamestate"].update_one({"device_id": req.device_id}, {"$set": {"is_completed": True}})
                # Update stats
                db["stats"].update_one({"device_id": req.device_id}, {"$inc": {"games_played": 1, "games_won": 1}})
                # best time update
                best = get_documents("stats", {"device_id": req.device_id}, limit=1)[0].get("best_time_seconds")
                if best is None or (updated_gs.get("elapsed_seconds", 0) < best):
                    db["stats"].update_one({"device_id": req.device_id}, {"$set": {"best_time_seconds": updated_gs.get("elapsed_seconds", 0)}})
            elif not done:
                db["stats"].update_one({"device_id": req.device_id}, {"$inc": {"games_played": 0}})

    # Return fresh state
    fresh = get_gamestate(req.device_id)
    fresh.pop("_id", None)
    return {"status": "ok", "gamestate": fresh, "conflicts": conflicts}

@app.post("/api/erase")
def erase_cell(req: EraseRequest):
    gs = get_gamestate(req.device_id)
    if not gs:
        raise HTTPException(status_code=404, detail="No game found")
    r, c = req.row, req.col
    if gs["fixed"][r][c]:
        raise HTTPException(status_code=400, detail="Cannot change a fixed cell")
    db["gamestate"].update_one({"device_id": req.device_id}, {"$set": {f"current.{r}.{c}": 0, f"notes.{r}.{c}": []}})
    fresh = get_gamestate(req.device_id)
    fresh.pop("_id", None)
    return {"status": "ok", "gamestate": fresh}

@app.post("/api/timer")
def update_timer(req: TimerUpdate):
    gs = get_gamestate(req.device_id)
    if not gs:
        raise HTTPException(status_code=404, detail="No game found")
    db["gamestate"].update_one({"device_id": req.device_id}, {"$set": {"elapsed_seconds": req.elapsed_seconds}})
    return {"status": "ok"}

@app.get("/api/stats/{device_id}")
def get_stats(device_id: str):
    st = get_documents("stats", {"device_id": device_id}, limit=1)
    if not st:
        return {"status": "empty"}
    st = st[0]
    st.pop("_id", None)
    return {"status": "ok", "stats": st}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
