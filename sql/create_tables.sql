-- enable extension
CREATE EXTENSION vector;

-- a dummy table
CREATE TABLE fruits(
   id SERIAL PRIMARY KEY,
   name VARCHAR NOT NULL
);

INSERT INTO fruits(name) 
VALUES('Orange');

-- hold embeddings
CREATE TABLE items (
   id bigserial PRIMARY KEY,
   page SMALLINT,
   embedding vector(384)
);