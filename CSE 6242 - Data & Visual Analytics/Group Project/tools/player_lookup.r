#--- INSTALL AND/OR LOAD  LIBRARIES
libraries  = c("data.table")


for(l in libraries){
	if(require(l, character.only = T)){
    print(l)
  }
  if(!require(l, character.only = T)){
    install.packages(l)
  }
}

url  <- "https://www.smartfantasybaseball.com/PLAYERIDMAPCSV" 
data <- fread(url, stringsAsFactors=FALSE)
out  <- data[,.(MLBID,MLBNAME)]
fwrite(out, "./data/player_lookup.csv")