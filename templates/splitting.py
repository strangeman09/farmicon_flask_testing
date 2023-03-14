import splitfolders

splitfolders.ratio("archive\dataset_", # The location of dataset
                   output="main train test", # The output location
                   seed=42, # The number of seed
                   ratio=(0.8,0.2), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )

