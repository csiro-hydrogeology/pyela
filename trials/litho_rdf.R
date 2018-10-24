library(rdflib)
litho = rdf_parse(doc = '/home/per202/Documents/Lithology/lithology.rdf')

sparql <- 'PREFIX ab: <http://resource.geolba.ac.at/lithology/> 

SELECT ?definition
WHERE
{ ab:22 http://www.w3.org/2004/02/skos/core#definition ?definition . }'

rdf_query(litho, sparql)
