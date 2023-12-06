import { writeFile, mkdir } from 'node:fs';
import dotenv from 'dotenv';
import { OpenAI } from "langchain/llms/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const question = 'В чем разница между параллельным и последовательным соединением?';
const resourceURL = 'https://alexgyver.ru/lithium_charging/';

dotenv.config();

new OpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });

const loader = new CheerioWebBaseLoader(resourceURL);
const logError = err => {
  if (err) console.log(err);
};

(async () => {
  const data = await loader.load();
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
  });
  const splitDocs = await textSplitter.splitDocuments(data);
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  
  const relevantDocs = await vectorStore.similaritySearch(question);
  const date = Date.now();
  mkdir('./answers', logError);
  mkdir('./vectorStore', logError);
  writeFile(`./vectorStore/${date}_vector.json`, JSON.stringify(vectorStore.memoryVectors), logError);
  writeFile(`./answers/${date}.json`, JSON.stringify(relevantDocs), logError);
})();
