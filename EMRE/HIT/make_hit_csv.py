import argparse
import os,sys
import sqlite3


def main():
    parser = argparse.ArgumentParser(description='Analyze logs.')
    parser.add_argument('-d', '--database', metavar='DATABASE', help='database file')
    parser.add_argument('-o', '--output', metavar='OUTPUT', help='output csv')

    args = parser.parse_args()

    database = args.database
    output = args.output
    
    if os.path.isfile(database):
        connection = sqlite3.connect(database)
        outfile = open(output,"w+")
        outfile.write("VIDEOA,DESCA,VIDEOB,DESCB,VIDEOC,DESCC,VIDEOD,DESCD,VIDEOE,DESCE\n")

        interval = list(range(1,6))
        
        with connection:
            cursor = connection.cursor()
        
            while(True):
                print(interval)
                cmd = "SELECT * FROM EMREVideoDBEntry where Id BETWEEN " + str(interval[0]) + " AND " + str(interval[-1])
                print(cmd)
                cursor.execute(cmd)
            
                results = cursor.fetchall()
                
                if len(results) == 0:
                    break
            
                row_list = []
                for result in results:
                    video_link = "https://s3.amazonaws.com/emre-videos/emre_vid/" + result[1] + ".mp4"
                    video_desc = '"'+result[5]+'"'
                    row_list.append(video_link + "," + video_desc)
                row = ",".join(row_list)
                outfile.write(row+"\n")
                print(row)
        
                interval = [i+5 for i in interval]
        
        connection.close()
    else:
        print("%s is not a file" % database)

if __name__ == "__main__":
	main()
