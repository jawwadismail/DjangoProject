import pymysql


class storedb(object):

    
    def store(self,F_NAME,F_SCORE):
        self.F_NAME=F_NAME
        self.F_SCORE=F_SCORE
        
        # Connect to the database
        connection = pymysql.connect(host='localhost',
                                    user='root',
                                    password='',
                                    db='cyberproject',
                                    charset='utf8mb4',
                                    cursorclass=pymysql.cursors.DictCursor)

        try:
            with connection.cursor() as cursor:
                # Create a new record
                sql = "INSERT INTO `tweets` (`name`, `score`) VALUES (%s, %f)"
                for i in range(len(F_NAME)):
                    cursor.execute(sql, (F_NAME[i],F_SCORE[i]))

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            connection.commit()

            with connection.cursor() as cursor:
                # Read a single record
                sql = "SELECT `name` FROM `tweets` WHERE `score`=%f"
                for i in range(len(F_NAME)):
                    cursor.execute(sql, (F_SCORE[i]))
            
                result = cursor.fetchone()
                print(result)
        finally:
            connection.close()


