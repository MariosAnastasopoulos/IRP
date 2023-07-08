# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 00:23:05 2023

@author: Marios
"""

import pymysql

class Create_Ticket:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
    
    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
                )
            self.cursor = self.connection.cursor()
            print('Connected to MySQL database')

        except pymysql.Error as e:
            print('Error connecting to MySQL:', e)        
        except pymysql.Error as e:
            print('Error connecting to MySQL:', e)        
    
    def get_latest_value(self):
    # Example function to retrieve the latest value from the database for the ticket id
        select_query = "SELECT MAX(ticket_code) FROM ticket"
        self.cursor.execute(select_query)
        latest_value = self.cursor.fetchone()[0]
        return latest_value
    
    def insert_ticket(self):
        try:
            # Get the latest entry in the database for a specific column
            latest_value = self.get_latest_value()

            # Increment the first entry in the data tuple by 1
            ticket_id= latest_value + 1
            
            insert_query = "INSERT INTO ticket (ticket_code, ticket_ticket_type_code\
                , ticket_ticket_priority_code, ticket_ticket_group_code,\
                ticket_ticket_status_code, ticket_subject, ticket_start_date,\
                ticket_due_date, ticket_desc, ticket_satellite, ticket_verified,\
                ticket_in_flag, ticket_out_flag, noc_ticket_services_code,\
                noc_ticket_issue_code, noc_ticket_operation_code)\
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            data = (ticket_id, 1, 5, 1, 1, "Wheel Unloading", 0, 0, "ab", 3, 0, 0, 0, 0, 0, 0)
            # Execute the insertion query with the modified data
            self.cursor.execute(insert_query, data)
            self.connection.commit()
            print('Data inserted to table ticket successfully')

        except pymysql.Error as e:
            print('Error writing to MySQL:', e)


    def insert_ticket_task(self, task_decription):
        try:
            ticket_id = self.get_latest_value()
            select_query = "SELECT MAX(ticket_task_code) FROM ticket_task"
            self.cursor.execute(select_query)
            task_id = self.cursor.fetchone()[0] + 1
            
            insert_query = "INSERT INTO ticket_task (ticket_task_code, ticket_task_ticket_code,\
                ticket_task_task_type_code, ticket_task_status_code, ticket_task_ticket_phase_code,\
                ticket_task_subject, ticket_task_start_date, ticket_task_due_date,\
                ticket_task_ins_date, ticket_task_desc)\
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            data = (task_id, ticket_id, 1, 1, 1, "Wheel Unloading", 0, 0, 0, task_decription)    
            # Execute the insertion query with the modified data
            self.cursor.execute(insert_query, data)
            self.connection.commit()
            print('Data inserted to table ticket_task successfully')
            
        except pymysql.Error as e:
            print('Error writing to MySQL:', e)

    def close_connection(self):
        if self.connection:
            self.connection.close()
            
            
database=Create_Ticket('127.0.0.1', 'root', 'Marios970511', 'irp-ticket')

database.connect()
database.insert_ticket()
database.insert_ticket_task("task 2")
database.close_connection()
