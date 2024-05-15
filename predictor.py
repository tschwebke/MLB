import matplotlib.pyplot as plt
import MySQLdb
import numpy as np
import pandas as pd
import scipy.stats as sps



def getSeasonData():
	'''
	Connects to the local database server which has been populated with data
	from the Lahman 2016 MLB database, available at
	http://www.seanlahman.com/baseball-archive/statistics/, and returns a pandas
	DataFrame object with season data for all MLB batters.
	'''

	connection = MySQLdb.connect(user = 'alexander', passwd = '',
		db = 'lahman2016')

	query = '''
		SELECT t1.playerID, CONCAT(master.nameFirst, ' ', master.nameLast) as
		       name, AB, H, BB, HBP, SF,

		       IFNULL((H / AB), 0) as avg,

		       IFNULL(((H + BB + HBP) / (AB + HBP + BB + SF)), 0) as obp

		  FROM (SELECT playerID,
		               CAST(SUM(AB) as UNSIGNED) as AB,
		               CAST(SUM(H) as UNSIGNED) as H,
		               CAST(SUM(2B) as UNSIGNED) as 2B,
		               CAST(SUM(3B) as UNSIGNED) as 3B,
		               CAST(SUM(HR) as UNSIGNED) as HR,
		               CAST(SUM(BB) as UNSIGNED) as BB,
		               CAST(SUM(HBP) as UNSIGNED) as HBP,
		               CAST(SUM(SF) as UNSIGNED) as SF
		          FROM batting
		         GROUP BY yearID, playerID
		       ) t1
		       INNER JOIN master
		       ON t1.playerID = master.playerID
		 ORDER BY playerID;
	'''

	df = pd.read_sql(query, connection)

	connection.close()

	return df



def computeSeasonNumbers(df):
	'''
	Takes a DataFrame with season data and returns a vector (with indices
	corresponding to the DataFrame indices) containing the year number of each
	player's seasons. That is, a player's first season is given the value 1, the
	second season is given the value 2, and so on. It is assumed that df is
	sorted by year.
	'''

	years = np.zeros((df.shape[0],), dtype = np.int16)

	lastPlayerID = ''
	year = 0

	for index, row in df.iterrows():
		if row.playerID == lastPlayerID:
			year += 1
		else:
			year = 1
			lastPlayerID = row.playerID

		years[index] = year

	return years



def abTestApprox(alphaA, betaA, alphaB, betaB):
	'''
	Returns the probability, for random variables A = Beta(alphaA, betaA) and
	B = Beta(alphaB, betaB), that A > B. Uses a normal distribution
	approximation to the beta distribution for computational efficiency.
	'''

	u1 = alphaA / (alphaA + betaA)
	u2 = alphaB / (alphaB + betaB)

	var1 = (alphaA * betaA) / (np.power(alphaA + betaA, 2) * \
		(alphaA + betaA + 1))
	var2 = (alphaB * betaB) / (np.power(alphaB + betaB, 2) * \
		(alphaB + betaB + 1))

	return sps.norm(u2 - u1, np.sqrt(var1 + var2)).cdf(0)



def betaPosterior(successes, total, priorA, priorB):
	'''
	Computes a generic posterior of a random variable described by a Beta
	distribution, where successes is the number of occurrences in total tests,
	and priorA and priorB are the parameters of the prior distribution.

	Returns, in succession, the posterior alpha, posterior beta, and posterior
	mean values.
	'''

	posteriorA = priorA + successes
	posteriorB = priorB + total - successes
	posteriorMean = posteriorA / (posteriorA + posteriorB)

	return posteriorA, posteriorB, posteriorMean



def avgPosterior(hits, atBats, priorA, priorB):
	'''
	Returns, in succession, the posterior alpha, posterior beta, and posterior
	mean values for batting average.
	'''

	return betaPosterior(hits, atBats, priorA, priorB)



def obpPosterior(hits, walks, hitByPitches, atBats, sacrificeFlies, priorA,
	priorB):
	'''
	Returns, in succession, the posterior alpha, posterior beta, and posterior
	mean values for on-base percentage.
	'''

	return betaPosterior(hits + walks + hitByPitches,
		atBats + walks + hitByPitches + sacrificeFlies, priorA, priorB)



def priorEstimate(data):
	'''
	Uses the providded data to fit a beta distribution with returned parameters
	alpha and beta.
	'''

	alpha, beta, floc, fscale = sps.beta.fit(data, 1, 10, floc = 0, fscale = 1)

	return alpha, beta



def similarity(alphaA, betaA, alphaB, betaB):
	'''
	Uses the probability comparison method to report a similarity between
	distributions A = Beta(alphaA, betaA) and B = Beta(alphaB, betaB) in the
	range [0, 0.5], where 0 indicates identical distributions.
	'''

	prob = abTestApprox(alphaA, betaA, alphaB, betaB)

	return np.abs(prob - 0.5)



def predictAVG(df, abThreshold, credibleInterval):
	'''
	Uses the seasons in df to generate a prior of batting average. The mean of
	the prior, as well as the bounds of the credible interval (size determined
	by credibleInterval, in the range (0,1)), are returned. The prior is
	generated using seasons featuring at least abThreshold ABs.
	'''

	seasons = df.loc[lambda df: df.AB >= abThreshold]

	# Pre-allocate the distribution vector
	distribution = np.zeros(int(np.sum(np.round(
		seasons.loc[:, 'weight'] * 100))))

	lastIndex = 0
	for index, row in seasons.iterrows():
		# The number of times an average is counted in the distribution is its
		# weight * 100
		numInsertions = int(round(row['weight'] * 100))
		distribution[lastIndex:lastIndex + numInsertions] = row['avg']

		lastIndex += numInsertions

	avgA, avgB, avgFloc, avgFscale = sps.beta.fit(distribution, 1, 10,
		floc = 0, fscale = 1)

	return (avgA - 1) / (avgA + avgB - 2), \
		sps.beta.ppf((1 - credibleInterval) / 2, avgA, avgB), \
		sps.beta.ppf(1 - ((1 - credibleInterval) / 2), avgA, avgB)



def predictOBP(df, abThreshold, credibleInterval):
	'''
	Uses the seasons in df to generate a prior of on-base percentage. The mean
	of the prior, as well as the bounds of the credible interval (size
	determined by credibleInterval, in the range (0,1)), are returned. The prior
	is generated using seasons featuring at least abThreshold ABs.
	'''

	seasons = df.loc[lambda df: df.AB >= abThreshold]

	# Pre-allocate the distribution vector
	distribution = np.zeros(int(np.sum(np.round(
		seasons.loc[:, 'weight'] * 100))))

	lastIndex = 0
	for index, row in seasons.iterrows():
		# The number of times an average is counted in the distribution is its
		# weight * 100
		numInsertions = int(round(row['weight'] * 100))
		distribution[lastIndex:lastIndex + numInsertions] = row['obp']

		lastIndex += numInsertions

	obpA, obpB, obpFloc, obpFscale = sps.beta.fit(distribution, 1, 10,
		floc = 0, fscale = 1)

	return (obpA - 1) / (obpA + obpB - 2), \
		sps.beta.ppf((1 - credibleInterval) / 2, obpA, obpB), \
		sps.beta.ppf(1 - ((1 - credibleInterval) / 2), obpA, obpB)



def plotPredictions(player, predictionsAVG, predictionsOBP, N):
	'''
	Plots the first N years of player data (average and on-base percentage), and
	previously generated predictions using matplotlib.
	'''

	fig, ax = plt.subplots(2, 1)
	
	ax[0].plot(player.index[:N] + 1, player[:N].loc[:, 'avg'], 'ko-')
	ax[0].plot([N, N + 1], [player.loc[:, 'avg'][N - 1],
		predictionsAVG.loc[:, 'mean'][N + 1]], 'k-')
	ax[0].plot(predictionsAVG.index, predictionsAVG.loc[:, 'mean'], 'ko--')
	ax[0].errorbar(predictionsAVG.index, predictionsAVG.loc[:, 'mean'],
		yerr = [predictionsAVG.loc[:, 'mean'] - predictionsAVG.loc[:, 'lower'],
			predictionsAVG.loc[:, 'upper'] - predictionsAVG.loc[:, 'mean']],
		color = 'grey', capsize = 15, linestyle = None)
	ax[0].set_title('%s AVG and OBP Prediction' % player.ix[0, 'name'])
	ax[0].grid()
	ax[0].set_xlabel('Year')
	ax[0].set_ylabel('Batting Average')

	ax[1].plot(player.index[:N] + 1, player[:N].loc[:, 'obp'], 'ko-')
	ax[1].plot([N, N + 1], [player.loc[:, 'obp'][N - 1],
		predictionsOBP.loc[:, 'mean'][N + 1]], 'k-')
	ax[1].plot(predictionsOBP.index, predictionsOBP.loc[:, 'mean'], 'ko--')
	ax[1].errorbar(predictionsOBP.index, predictionsOBP.loc[:, 'mean'],
		yerr = [predictionsOBP.loc[:, 'mean'] - predictionsOBP.loc[:, 'lower'],
			predictionsOBP.loc[:, 'upper'] - predictionsOBP.loc[:, 'mean']],
		color = 'grey', capsize = 15, linestyle = None)
	ax[1].grid()
	ax[1].set_xlabel('Year')
	ax[1].set_ylabel('On-Base Percentage')	

	plt.show()



def main():
	'''
	The main program. Uses data from the Lahman 2016 database to predict the
	future performance of a player using Bayesian methods. The player's AVG and
	OBP are compared to other players during the first N years of their careers,
	and weightings are drawn form these comparisons to create predictions for
	subsequent years.

	TODO: Add SLG and WAR predictions.
	'''

	# Change these parameters. The playerToTest should be the player's playerID,
	# which is the player's ID on baseball-reference.com
	playerToTest = 'troutmi01'
	N = 4
	abThreshold = 100
	maxSeasons = 15
	credibleInterval = 0.8

	

	df = getSeasonData()

	df.loc[:, 'yearID'] = computeSeasonNumbers(df)

	# To generate a prior estimate, use batter seasons with a minimum AB
	dfFiltered = df.loc[lambda df: df.AB >= abThreshold]
	avgAlpha, avgBeta = priorEstimate(dfFiltered.loc[:, 'avg'])
	obpAlpha, obpBeta = priorEstimate(dfFiltered.loc[:, 'obp'])

	df.loc[:, 'avg_alpha'], df.loc[:, 'avg_beta'], df.loc[:, 'avg_eb'] = \
		avgPosterior(df.loc[:, 'H'], df.loc[:, 'AB'], avgAlpha, avgBeta)

	df.loc[:, 'obp_alpha'], df.loc[:, 'obp_beta'], df.loc[:, 'obp_eb'] = \
		obpPosterior(df.loc[:, 'H'], df.loc[:, 'BB'], df.loc[:, 'HBP'],
			df.loc[:, 'AB'], df.loc[:, 'SF'], obpAlpha, obpBeta)



	# Select the player to test and remove those seasons from df
	playerRows = df.loc[lambda df: df.playerID == playerToTest].index

	if playerRows.shape[0] < N:
		# We don't have the required number of seasons for comparison
		raise Exception('Not enough seasons for this player')

	player = df.iloc[playerRows].copy().reset_index(drop = True)
	df.drop(playerRows, inplace = True)

	# Remove players who have fewer than N seasons of data
	df = df[df.groupby('playerID').playerID.transform(len) >= N]




	# Compute the similarities of AVGs and OBPs for all players against the
	# selected player
	for i in range(N):
		avgAlphas = df.loc[lambda df: df.yearID == i + 1, 'avg_alpha'].copy()
		avgBetas = df.loc[lambda df: df.yearID == i + 1, 'avg_beta'].copy()

		playerAVGAlpha = player.ix[i, 'avg_alpha']
		playerAVGBeta = player.ix[i, 'avg_beta']

		obpAlphas = df.loc[lambda df: df.yearID == i + 1, 'obp_alpha'].copy()
		obpBetas = df.loc[lambda df: df.yearID == i + 1, 'obp_beta'].copy()

		playerOBPAlpha = player.ix[i, 'obp_alpha']
		playerOBPBeta = player.ix[i, 'obp_beta']

		df.loc[lambda df: df.yearID == i + 1, 'avg_sim'] = similarity(
			avgAlphas, avgBetas, playerAVGAlpha, playerAVGBeta)

		df.loc[lambda df: df.yearID == i + 1, 'obp_sim'] = similarity(
			obpAlphas, obpBetas, playerOBPAlpha, playerOBPBeta)



	# Take a copy of player seasons that fall within the first N years of
	# players' careers
	firstNYears = df.loc[lambda df: df.yearID <= N,
		['playerID', 'name', 'avg_sim', 'obp_sim']].copy()

	# Find the mean similarity for AVG and OBP (separately) for each player
	firstNYears = firstNYears.groupby(['playerID', 'name']).mean().reset_index()

	firstNYears.loc[:, 'mean_sim'] = (firstNYears.loc[:, 'avg_sim'] + \
		firstNYears.loc[:, 'obp_sim']) / 2



	# The weighting function that uses the mean similarity across N years and
	# both AVG and OBP metrics: w = max(0.5 - s^4, 0). This returns a weight in
	# the range [0, 1].
	firstNYears.loc[:, 'weight'] = np.maximum(np.power(0.5 - \
		firstNYears.loc[:, 'mean_sim'], 4), 0)


	# The weights are inserted into the original DataFrame (for each year of a
	# player's career)
	df = pd.merge(df, firstNYears.loc[:, ['playerID', 'weight']],
		on = 'playerID')

	predictionsAVG = pd.DataFrame([], columns = ['mean', 'lower', 'upper'])
	predictionsOBP = pd.DataFrame([], columns = ['mean', 'lower', 'upper'])

	for i in range(N + 1, maxSeasons + 1):
		predictionsAVG.loc[i] = list(predictAVG(
			df.loc[lambda df: df.yearID == i], abThreshold, credibleInterval))

		predictionsOBP.loc[i] = list(predictOBP(
			df.loc[lambda df: df.yearID == i], abThreshold, credibleInterval))

	plotPredictions(player, predictionsAVG, predictionsOBP, N)



if __name__ == '__main__':
	main()