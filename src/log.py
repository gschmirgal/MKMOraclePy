import datetime
from src.db import dbMkmPy

class log:
    db = dbMkmPy()
    statuses = {}
    statusesFlip = {}
    id = None
    dateImport = datetime.datetime.now()
    dateImportFile = None
    dateData = None
    status = "ongoing"
    task = None
    
    def __init__(self):
        db = dbMkmPy()
        self.statuses = self.getStatuses()
        self.statusesFlip = {v: k for k, v in self.statuses.items()}

        self.tasks = self.getTasks()
        self.tasksFlip = {v: k for k, v in self.tasks.items()}

    def getStatuses(self):
        datadb = self.db.query("SELECT * FROM logsteps WHERE 1")
        data = {}
        for row in datadb:
            data[row['id']] = row['step']
        return data

    def getTasks(self):
        datadb = self.db.query("SELECT * FROM taskstypes WHERE 1")
        data = {}
        for row in datadb:
            data[row['id']] = row['task']
        return data

    def createLogEntry(self, task = None):

        if task not in self.tasks.values():
            raise ValueError(f"Invalid task: {task}")
        
        self.task = task

        sql = f"INSERT INTO logs_oracle (date, idStep, idTask) VALUES ('{self.dateImport.strftime('%Y-%m-%d %H:%M:%S')}', '{self.statusesFlip[self.status]}', '{self.tasksFlip[self.task] }')"
        self.db.query(sql)

        self.id = self.db.get1value("SELECT LAST_INSERT_ID()")
        return self.id

    def setStatus(self, status):
        if status not in self.statuses.values():
            raise ValueError(f"Invalid status: {status}")
        
        self.status = status

        sql = f"UPDATE logs_oracle SET idStep = '{self.statusesFlip[self.status]}' WHERE id = {self.id}"
        self.db.query(sql)