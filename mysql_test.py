import pandas as pd
from sqlalchemy import create_engine

USER = "root"
PWD  = "test"
HOST = "127.0.0.1"
PORT = 3306
DB   = "exercise_data"

engine = create_engine(
    f"mysql+pymysql://{USER}:{PWD}@{HOST}:{PORT}/{DB}",
    connect_args={"charset": "utf8mb4"},  # 클라이언트는 utf8mb4
    future=True
)

df = pd.read_csv("./KS_PTDRCTOR_WRTNG_INFO_202507.csv")
df = df.head()
print(df.head())

try:
    df.to_sql("exercise_data", con=engine, if_exists="append", index=False)
    print("insert success(finish)")
except Exception as e:
    
    print("DRIVER ORIG:", e.orig)
    print()
    print("SQL:", e.statement)
    print()
    print("PARAMS:", getattr(e, "params", None))
    print()
    if hasattr(e.orig, "args") and e.orig.args:
        print("MYSQL CODE:", e.orig.args[0])




# utf8mb4_uca1400_ai_ci