##-------------------------------------------------------------
## SETUP	
##-------------------------------------------------------------
# root 	   		<- "C:/Users/sbail/OneDrive/Documents/projects/baseball/pitch_prediction"
root 	   		<- getwd()
data_path  		<- paste(root,"data",sep="/")
programs_path   <- paste(root,"programs",sep="/") 
output_path     <- paste(root,"output",sep="/")

#--- INSTALL AND/OR LOAD  LIBRARIES
libraries <- c("iterators", "doParallel","foreach", 
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
## KEEP SUBSET OF COLUMNS FROM ORIGINAL FILES
##-------------------------------------------------------------
keep_cols <- c("pitch_type","game_year","game_pk","pitcher","batter","at_bat_number",
			   "pitch_number","release_speed","release_pos_x","release_pos_y",
  			   "release_pos_z","description","events","zone","stand","p_throws","type",
  			   "bb_type","balls","strikes","pfx_x","pfx_z","plate_x","plate_z","on_3b",
  			   "on_2b","on_1b","outs_when_up","inning","inning_topbot","vx0","vy0","vz0",
  			   "ax","ay","az","sz_top","sz_bot","hit_distance_sc","launch_speed",
  			   "launch_angle","effective_speed","release_spin_rate","release_extension",
  			   "estimated_ba_using_speedangle","estimated_woba_using_speedangle",
  			   "woba_value","woba_denom","babip_value","iso_value","launch_speed_angle",
  			   "bat_score","fld_score","post_bat_score","post_fld_score",
  			   "if_fielding_alignment","of_fielding_alignment")

##-------------------------------------------------------------
## DEFINE PITCH TYPES
##-------------------------------------------------------------
## Keep only pitch types listed below (listed in order of frequency in 2018)
# Abbrvs correspond to: 4-seam Fastball, Slider, 2-seam Fastball, Changeup,
# Curveball, Sinker, Cut Fastball (aka Cutter), Knuckle-curve, 
# Split-Fingered Fastball (aka Splitter), and Knuckle Ball
pitch_types_list <- c("FF","SL","FT","CH","CU","SI","FC","KC","FS","KN")

##---------------------------------------
## LOAD EACH SEASON, PREPARE, COMBINE
##---------------------------------------
files <- list.files("./data/single_season_files", pattern=".csv")

registerDoParallel(cores=3)
combined_list <- foreach(i = files) %dopar% {
	data <- data.table::fread(paste0("./data/single_season_files/",i), select= keep_cols, stringsAsFactors=FALSE)
	return(data)
	rm(data)
}

data <- rbindlist(combined_list)
rm(combined_list)

##---------------------------------------
## CREATE BASIC NEW VARIABLES (No lags, these vars will be available for every pitch regardless of situation)
##---------------------------------------
## Turn baserunner variables into binary 0-1 flags
data[, BR_EMPTY := ifelse(is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]
data[, BR_1B    := ifelse(!is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]
data[, BR_1B_2B := ifelse(!is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]
data[, BR_1B_3B := ifelse(!is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]
data[, BR_2B    := ifelse(!is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]
data[, BR_2B_3B := ifelse(!is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]
data[, BR_3B    := ifelse(!is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]
data[, BR_FULL  := ifelse(!is.na(on_1b) & is.na(on_2b) & is.na(on_3b),1,0)]	

## Create "Count" binary 0-1 flags 
# First, convert pitches that occurred with 4 balls to 3 balls (it's impossible for any pitch to be thrown with 4 balls in the count) 
# There are only 8 instances of this in 2018, so I'm pretty sure they are data errors
data[balls == 4, balls := 3]
data[, zero_zero_count  := ifelse(balls == 0 & strikes == 0,1,0)]
data[, zero_one_count   := ifelse(balls == 0 & strikes == 1,1,0)]
data[, zero_two_count   := ifelse(balls == 0 & strikes == 2,1,0)]
data[, one_zero_count   := ifelse(balls == 1 & strikes == 0,1,0)]
data[, one_one_count    := ifelse(balls == 1 & strikes == 1,1,0)]
data[, one_two_count    := ifelse(balls == 1 & strikes == 2,1,0)]
data[, two_zero_count   := ifelse(balls == 2 & strikes == 0,1,0)]
data[, two_one_count    := ifelse(balls == 2 & strikes == 1,1,0)]
data[, two_two_count    := ifelse(balls == 2 & strikes == 2,1,0)]
data[, three_zero_count := ifelse(balls == 3 & strikes == 0,1,0)]
data[, three_one_count  := ifelse(balls == 3 & strikes == 1,1,0)]
data[, three_two_count  := ifelse(balls == 3 & strikes == 2,1,0)]
	
## Recode extra innings and create dummies
data[inning > 9, inning := 10]
uinning <- sort(unique(data$inning))
data[, (paste0("inning",uinning)) := 0]

for(x in uinning) {
  set(data, i = which(data[["inning"]] == x), 
  		  			  j = paste0("inning",x), value = 1)
}

## Calculate the score differential at time of pitch
data[, score_diff := fld_score - bat_score]

# Get game pitch-count for specific pitcher
setkey(data,game_pk,pitcher,at_bat_number,pitch_number)
key_pn <- c("game_pk","pitcher")
data[, game_pitch_number := seq(.N), by=key_pn]

# Get where current batter hits in the batting order and create dummies
k <- c("game_pk","inning","inning_topbot","at_bat_number")
setkeyv(data, cols = k)
unique_order <- unique(data[, k, with=FALSE])
unique_order[, atbat_team := seq_len(.N), by=c("game_pk","inning_topbot")]
unique_order[, batting_order := atbat_team %% 9]
unique_order[batting_order == 0, batting_order := 9]
batorders <- unique(unique_order$batting_order)
unique_order[, (paste0("batting_order",batorders)) := 0]

for(x in batorders) {
  set(unique_order, i = which(unique_order[["batting_order"]] == x), 
  		  			j = paste0("batting_order",x), value = 1)
}

data <- merge(data, unique_order, by= k)
rm(unique_order)

##---------------------------------------
## CREATE LAG VARIABLES FOR PITCHERS 
##---------------------------------------
# Get release_speed for 3 previous pitches
setkey(data,game_pk,pitcher,game_pitch_number)
data[,sprintf("release_speedlag%d",1:3):=shift(release_speed,1:3,type="lag"),by=key_pn]

for (i in 2:3){ # These are averages over last 2 and 3 pitches
	var_name  <- paste0("avg",i,"_release_speed")
	data[,(var_name):=rowMeans(.SD),.SDcols=sprintf("release_speedlag%d",1:i)]
}
# Get horizontal and vertical location lags
data[,sprintf("plate_xlag%d",1:3):=shift(plate_x,1:3,type="lag"),by=key_pn]
data[,sprintf("plate_zlag%d",1:3):=shift(plate_z,1:3,type="lag"),by=key_pn]

for (i in 2:3){ # These are averages over last 2 and 3 pitches
	var_name  <- paste0("avg",i,"_plate_x")
	var_name2 <- paste0("avg",i,"_plate_z")
	data[,(var_name):=rowMeans(.SD),.SDcols=sprintf("plate_xlag%d",1:i)]
	data[,(var_name2):=rowMeans(.SD),.SDcols=sprintf("plate_zlag%d",1:i)]
}
# Get horizontal and vertical movement lags 
data[,sprintf("pfx_xlag%d",1:3):=shift(pfx_x,1:3,type="lag"),by=key_pn]
data[,sprintf("pfx_zlag%d",1:3):=shift(pfx_z,1:3,type="lag"),by=key_pn]

for (i in 2:3){ # These are averages over last 2 and 3 pitches
	var_name  <- paste0("avg",i,"_pfx_x")
	var_name2 <- paste0("avg",i,"_pfx_z")
	data[,(var_name):=rowMeans(.SD),.SDcols=sprintf("pfx_xlag%d",1:i)]
	data[,(var_name2):=rowMeans(.SD),.SDcols=sprintf("pfx_zlag%d",1:i)]
}
# Combine outcomes & then
# get outcomes of 3 previous pitches
data[description == '', description := NA]
swing_strike <- c("swinging_strike","swinging_strike_blocked","missed_bunt","unknown_strike","swinging_pitchout")
fouls   <- c("foul","foul_bunt","foul_tip","foul_pitchout")
pballs   <- c("ball","blocked_ball","intent_ball","pitchout")
contact <- c("hit_into_play","hit_into_play_no_out","hit_into_play_score",
			 "pitchout_hit_into_play","pitchout_hit_into_play_no_out","pitchout_hit_into_play_score") 

data[description %in% swing_strike,  outcome := "swinging_strike"]
data[description == "called_strike", outcome := "called_strike"]
data[description %in% pballs,        outcome := "ball"]
data[description %in% fouls,         outcome := "foul"]
data[description == "hit_by_pitch",  outcome := "hit_by_pitch"]
data[description %in% contact,       outcome := events]

data[,sprintf("outcomelag%d",1:3):=shift(outcome,1:3,type="lag"),by=key_pn]

# Get pitch type of 3 previous pitches
data[,sprintf("pitch_typelag%d",1:3):=shift(pitch_type,1:3,type="lag"),by=key_pn]

##-------------------------------------------------------------
## 	Cleanup wOBA Denominator Variable
##-------------------------------------------------------------
event_denom <-  c("field_out","strikeout","walk","double",                   
 				  "home_run","single","double_play","triple",                   
 				  "force_out","grounded_into_double_play","hit_by_pitch",
 				  "sac_fly","fielders_choice_out","field_error",
 				  "strikeout_double_play","fielders_choice","triple_play",
 				  "sac_fly_double_play") 

data[events %in% event_denom, woba_denom := 1]

##-------------------------------------------------------------
## 	ONLY KEEP OBSERVATIONS WITH CLASSIFIED PITCH TYPES
##-------------------------------------------------------------
data <- data[pitch_type %in% pitch_types_list,]

##---------------------------------------
## OUTPUT ALL PITCHES COMBINED
##---------------------------------------
setkey(data,game_year,game_pk,pitcher,at_bat_number,pitch_number)
fwrite(data,"./data/All_Pitches.csv", na=NA)