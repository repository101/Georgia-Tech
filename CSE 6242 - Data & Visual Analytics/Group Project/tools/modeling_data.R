##-------------------------------------------------------------
## SETUP	
##-------------------------------------------------------------
# root 	   		<- "C:/Users/sbail/OneDrive/Documents/projects/baseball/pitch_prediction"
root 	   		<- getwd()
data_path  		<- paste(root,"data",sep="/")
programs_path   <- paste(root,"programs",sep="/") 
output_path     <- paste(root,"output",sep="/")


#--- INSTALL AND/OR LOAD  LIBRARIES
libraries  = c("iterators", "doParallel","foreach", 
  			   "snow","data.table")


for(l in libraries){
	if(require(l, character.only = T)){
    print(l)
  }
  if(!require(l, character.only = T)){
    install.packages(l)
  }
}

##-------------------------------------------------------------
## 	READ DATA
##-------------------------------------------------------------
keep_cols <- c("pitch_type","game_year","game_pk","pitcher","batter",
  			   "pitch_number","game_pitch_number","outcome",
  			   "stand","outs_when_up","inning1","inning2","inning3","inning4","inning5",
  			   "inning6","inning7","inning8","inning9","inning10",
  			   "BR_EMPTY","BR_1B", "BR_1B_2B","BR_1B_3B","BR_2B","BR_2B_3B","BR_3B",
  			   "BR_FULL","zero_zero_count","zero_one_count","zero_two_count","one_zero_count",
  			   "one_one_count","one_two_count","two_zero_count","two_one_count","two_two_count",   
			   "three_zero_count","three_one_count","three_two_count","score_diff",
			   "batting_order1","batting_order2","batting_order3","batting_order4",
			   "batting_order5","batting_order6","batting_order7","batting_order8","batting_order9",
			   "release_speedlag1","release_speedlag2","release_speedlag3","avg2_release_speed","avg3_release_speed",             
			   "plate_xlag1","plate_xlag2","plate_xlag3","plate_zlag1","plate_zlag2","plate_zlag3",                    
			   "avg2_plate_x","avg2_plate_z","avg3_plate_x","avg3_plate_z","pfx_xlag1","pfx_xlag2",                      
			   "pfx_xlag3","pfx_zlag1","pfx_zlag2","pfx_zlag3","avg2_pfx_x","avg2_pfx_z",                     
			   "avg3_pfx_x","avg3_pfx_z","outcomelag1","outcomelag2","outcomelag3",                    
			   "pitch_typelag1","pitch_typelag2","pitch_typelag3","woba_value","woba_denom")

data <- fread("./data/All_Pitches.csv", select = keep_cols, stringsAsFactors=FALSE)

##-------------------------------------------------------------
## 	AGGREGATE BY BATTER SEASON AND PITCH TYPE
## 		- WOBA -- Weighted On Base Average (measure of batter skill)
##		- Swinging Strike% -- Percentage of pitches when batter swings and misses
##-------------------------------------------------------------
by_key <- c("game_year","batter","pitch_type")
batter_woba <- data[, { woba_aggnum = sum(woba_value, na.rm=TRUE);
						woba_aggdenom =  sum(woba_denom, na.rm=TRUE);
						woba = woba_aggnum/woba_aggdenom;
						list(woba=woba)},
						by = by_key]

woba_wide <- reshape(batter_woba, idvar = c("game_year","batter"), 
								  timevar = "pitch_type", 
								  direction = "wide")

batter_swstrike <- data[, { swstrike = length(batter[which(outcome == "swinging_strike")]);
						    swstrike_pct = swstrike/.N*100; 
						    list(swstrike_pct=swstrike_pct)},
						    by = by_key]

swstrike_wide <- reshape(batter_swstrike, idvar = c("game_year","batter"), 
									      timevar = "pitch_type", 
									      direction = "wide")

##-------------------------------------------------------------
## 	Get avg wOBA and Swinging Strike rate by Pitch Type for last 2 seasons
##-------------------------------------------------------------
# setkey(woba_wide,batter,game_year)
# for (x in grep)
# woba_wide[,sprintf("%d",1:3):=shift(pfx_x,1:3,type="lag"),by=key_pn]


# batter_season[, lag_woba := shift(woba,1,type="lag"),by=c("batter","pitch_type")]
# batter_season[, lag_swstrike_pct := shift(woba,1,type="lag"),by=c("batter","pitch_type")]
# batter_season[, woba_avg := (woba+lag_woba)/2]
# batter_season[, swstrike_pct_avg := (swstrike_pct+lag_swstrike_pct)/2]

##-------------------------------------------------------------
## 	Merge with main data
##-------------------------------------------------------------
woba_wide[, merge_year := game_year + 1]
woba_wide[, game_year  := NULL]
swstrike_wide[, merge_year := game_year + 1]
swstrike_wide[, game_year  := NULL]

data_m <- merge(data,woba_wide, 
					by.x = c("game_year","batter"),
					by.y = c("merge_year","batter"),
					all.x = TRUE)
rm(data)
rm(woba_wide)

data_m <- merge(data_m,swstrike_wide, 
					by.x = c("game_year","batter"),
					by.y = c("merge_year","batter"),
					all.x = TRUE)
rm(swstrike_wide)


##-------------------------------------------------------------
## 	KEEP PITCHERS WHO HAVE THROWN AT LEAST 3K PITCHES SINCE 2008 AND ALSO PITCHED IN 2018
##-------------------------------------------------------------
agg_pitches <- data_m[,list(num_pitches = .N), by=c("pitcher")]
keep_pitchers_3K <- agg_pitches[num_pitches >= 3000,]$pitcher
keep_pitchers_2018 <- unique(data_m[game_year == 2018,]$pitcher)
keep_pitchers <- intersect(keep_pitchers_3K,keep_pitchers_2018)

out <- data_m[pitcher %in% keep_pitchers & 
			 	    !is.na(batter) &
			 	    !is.na(release_speed) &
			 	    !is.na(pfx_x) &
			 	    !is.na(pfx_z) &
			 	    !is.na(plate_x) &
			 	    !is.na(plate_z),]

out[, woba_value := NULL]
out[, woba_denom := NULL]
out[, game_pk    := NULL]
out[, outcome    := NULL]


rm(data_m,agg_pitches)
fwrite(out,"./data/Modeling_data.csv", na=NA)
