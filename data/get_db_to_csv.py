from database import EMSDatabase
db = EMSDatabase()
db.export_csv("training_data.csv")   # 81 columns, one row per stimulation