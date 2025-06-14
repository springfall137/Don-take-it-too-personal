import { MongoClient } from 'mongodb'
const url = 'mongodb+srv://admin:qwer1234@cluster123.z8hpxko.mongodb.net/forum?retryWrites=true&w=majority&appName=Cluster123'
const options = { useNewUrlParser: true }
let connectDB

if (process.env.NODE_ENV === 'development') {
    if (!global._mongo) {
        global._mongo = new MongoClient(url, options).connect()
    }
    connectDB = global._mongo
} else {
    connectDB = new MongoClient(url, options).connect()
}

export { connectDB }

// const client = await MongoClient.connect('mongodb+srv://admin:qwer1234@cluster123.z8hpxko.mongodb.net/?retryWrites=true&w=majority&appName=Cluster123"', { useNewUrlParser: true })

// export {client}
