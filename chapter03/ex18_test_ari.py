from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score

clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]

print "Accuracy: {:.2f}".format(accuracy_score(clusters1, clusters2))
print "ARI: {:.2f}".format(adjusted_rand_score(clusters1, clusters2))
print "mi: {:.2f}".format(mutual_info_score(clusters1, clusters2))
print "nmi: {:.2f}".format(normalized_mutual_info_score(clusters1, clusters2))
