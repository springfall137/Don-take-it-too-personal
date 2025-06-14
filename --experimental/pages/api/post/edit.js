import { ObjectId } from "mongodb"
import { connectDB } from "@/util/database"

export default async function handler(request, response) {

    if (request.method == 'POST') {
        // console.log(request.body)
        let change = {title:request.body.title, content:request.body.content}
        const db = (await connectDB).db('forum')
        let result = await db.collection('post').updateOne(
            { _id : new ObjectId(request.body._id) },
            {$set : change}
        )
        response.redirect(302, '/list')
    }
}
