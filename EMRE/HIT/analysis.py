import argparse
import os,sys
import sqlite3
import csv
import itertools
import pprint
import statistics as stats

def main():
    parser = argparse.ArgumentParser(description='Analyze logs.')
    parser.add_argument('-d', '--database', metavar='DATABASE', help='database file')
    parser.add_argument('-b', '--batch', metavar='BATCH', help='batch data csv')
    parser.add_argument('-p', '--params', metavar='PARAMS', nargs='+', help='parameters to search')
    parser.add_argument('-m', '--mode', action='store_true', help='eval relative to mode')

    args = parser.parse_args()

    database = args.database
    batch = args.batch
    params = args.params
    mode = args.mode
    
    for i in range(len(params)):
        if "=" in params[i]:
            if not params[i][params[i].index("=")+1].isnumeric():
                params[i] = "='".join(params[i].rsplit("=",1))
                params[i] += "'"
                params[i] = params[i].replace("''","'")
    
    print(params)

    pp = pprint.PrettyPrinter(indent=4)
    
    batchdata = []
    
    obj_color = {
        "red_block1" : "red",
        "purple_block3" : "purple",
        "block4" : "green",
        "green_block5" : "green",
        "block6" : "red",
        "block7" : "purple"
    }
    
    counts = {}

    for i in range(1,len(params)+1):
        for key in [j for j in itertools.combinations(params, i)]:
            counts[" AND ".join(list(key))] = 0

    rank_counts = {}

    if mode:
        for i in range(len(list(counts.keys()))):
            rank_counts[list(counts.keys())[i] + " AND R=E (-2)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=D (-1)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=C (0)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=B (+1)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=A (+2)"] = 0
    else:
        for i in range(len(list(counts.keys()))):
            rank_counts[list(counts.keys())[i] + " AND R=E (1)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=D (2)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=C (3)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=B (4)"] = 0
            rank_counts[list(counts.keys())[i] + " AND R=A (5)"] = 0

    print(counts)

    hit_results = []

    if os.path.isfile(batch):
        file = open(batch)
        i = 0
        for entry in csv.reader(file):
            if entry[0] != "HITId":
                vidA = entry[27].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankA = int(entry[37])
                vidB = entry[29].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankB = int(entry[38])
                vidC = entry[31].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankC = int(entry[39])
                vidD = entry[33].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankD = int(entry[40])
                vidE = entry[35].replace("https://s3.amazonaws.com/emre-videos/emre_vid/","").replace(".mp4","")
                rankE = int(entry[41])
                
                hit_results.append([vidA,vidB,vidC,vidD,vidE,
                                  rankA,rankB,rankC,rankD,rankE])
                
                i += 1
    else:
        print("%s is not a file" % batch)
        exit()

    hit_results = hit_results
    #hit_results = hit_results[:8]
    for result in hit_results:
        print(result)

    if os.path.isfile(database):
        connection = sqlite3.connect(database)
        cursor = connection.cursor()
        
        for key in counts:
            #print(key)
            cmd = "SELECT FilePath FROM EMREVideoDBEntry WHERE " + key
            #cmd = "SELECT FilePath FROM EMREVideoDBEntry WHERE Id >= 111 AND Id <= 115"
            print(cmd)
            cursor.execute(cmd)
            results = [r[0] for r in cursor.fetchall()]
            print(results)
            counts[key] += 8*len(results)
        
            for filepath in results:
                vid_rankings = [hit_result for hit_result in hit_results if filepath in hit_result]
                for task in vid_rankings:
                    print(task)
                    rankings = [(-r+6 if task[5] > task[9] else r) for r in task[5:]]
                    if mode:
                        rankings = [r-stats.median(rankings) for r in rankings]
                        print(rankings)
                        rank_counts[key + " AND R=E (-2)"] += (rankings[task.index(filepath)] == -2)
                        rank_counts[key + " AND R=D (-1)"] += (rankings[task.index(filepath)] == -1)
                        rank_counts[key + " AND R=C (0)"] += (rankings[task.index(filepath)] == 0)
                        rank_counts[key + " AND R=B (+1)"] += (rankings[task.index(filepath)] == 1)
                        rank_counts[key + " AND R=A (+2)"] += (rankings[task.index(filepath)] == 2)
                    else:
                        rank_counts[key + " AND R=E (1)"] += (rankings[task.index(filepath)] == 1)
                        rank_counts[key + " AND R=D (2)"] += (rankings[task.index(filepath)] == 2)
                        rank_counts[key + " AND R=C (3)"] += (rankings[task.index(filepath)] == 3)
                        rank_counts[key + " AND R=B (4)"] += (rankings[task.index(filepath)] == 4)
                        rank_counts[key + " AND R=A (5)"] += (rankings[task.index(filepath)] == 5)
        #print(results)
        
        #P(Rank=5|Modality=E) = C(R=5,M=E)/C(M=E)
        #P(Rank=5|Modality=E,O=red_block1) = C(R=5,M=E,O=1)/C(M=E,O=1)


        pp.pprint(counts)
        pp.pprint(rank_counts)

        connection.close()
    else:
        print("%s is not a file" % database)

if __name__ == "__main__":
	main()
