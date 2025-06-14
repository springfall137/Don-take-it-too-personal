import { connectDB } from "@/util/database";
import { ObjectId } from "mongodb";

export default async function handler(rq, rp) {
    console.log(rq.query)
    const db = (await connectDB).db('forum')
    let aaa = await db.collection('comment').find({ parent : new ObjectId(rq.query.id)}).toArray()
    rp.status(200).json(aaa)
}
