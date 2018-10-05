library(rdflib)
litho = rdf_parse(doc = '/home/per202/Documents/Lithology/lithology.rdf')




parql <-
  'SELECT  ?Species ?Sepal_Length ?Sepal_Width ?Petal_Length  ?Petal_Width
WHERE {
 ?s <iris:Species>  ?Species .
 ?s <iris:Sepal.Width>  ?Sepal_Width .
 ?s <iris:Sepal.Length>  ?Sepal_Length . 
 ?s <iris:Petal.Length>  ?Petal_Length .
 ?s <iris:Petal.Width>  ?Petal_Width 
}'


sparql <- 'PREFIX ab: <http://resource.geolba.ac.at/lithology/> 

SELECT ?definition
WHERE
{ ab:22 http://www.w3.org/2004/02/skos/core#definition ?definition . }'

rdf_query(litho, sparql)
